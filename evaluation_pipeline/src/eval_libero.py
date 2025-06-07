from functools import partial
import os
import json
import jax
import numpy as np
from absl import logging
from pathlib import Path
from octo.model.octo_model import OctoModel
from octo.utils.train_callbacks import supply_rng
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from octo.utils.gym_wrappers import HistoryWrapper, RHCWrapper
from octo.libero.libero_utils import LiberoGymWrapper, normalize_gripper_action, invert_gripper_action
from eval_utils import load_test_cases, save_results

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm


def evaluate_task(cfg: DictConfig, model, task_name, task_suite_name, instruction, related_task=None):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    task_id = [task.name for task in task_suite.tasks].index(task_name)
    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": cfg.camera_heights,
        "camera_widths": cfg.camera_widths,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()

    max_episode_len = cfg["libero"][task_suite_name]["max_episode_len"]
    env = LiberoGymWrapper(
        env, camera_height=cfg.camera_heights, camera_width=cfg.camera_widths, max_episode_len=max_episode_len
    )
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=4)

    with open("/home/artur/octo_checkpoints/libero_100/subtasks_mix_2/dataset_statistics.json", "r") as f:
        dataset_statistics = json.load(f)

    if task_suite_name == "libero_90":
        dataset_name = "libero_90_original"
    elif task_suite_name == "libero_10":
        dataset_name = "libero_10_original_no_noops"
    elif task_suite_name == "libero_test":
        dataset_name = "libero_10_original_no_noops"
    else:
        dataset_name = "libero_90_original"

    for k, v in dataset_statistics[dataset_name]["action"].items():
        dataset_statistics[dataset_name]["action"][k] = np.array(v)

    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=dataset_statistics[dataset_name]["action"],
        ),
    )

    if task_suite_name == "libero_90" or task_suite_name == "libero_10":
        initial_states = task_suite.get_task_init_states(task_id)
    else:
        libero_90_suite = benchmark_dict["libero_90"]()
        libero_10_suite = benchmark_dict["libero_10"]()
        try:
            task_id = [task.name for task in libero_90_suite.tasks].index(related_task)
            initial_states = libero_90_suite.get_task_init_states(task_id)
        except ValueError as e:
            task_id = [task.name for task in libero_10_suite.tasks].index(related_task)
            initial_states = libero_10_suite.get_task_init_states(task_id)

    # running rollouts
    episode_returns = []
    num_subtasks = len(env.env.env.env.parsed_problem["goal_state"])
    for i in tqdm(range(cfg.num_rollouts)):
        env.reset()

        # if task_suite_name == "libero_90" or task_suite_name == "libero_10":
        obs = env.set_init_state(initial_states[i])

        language_instruction = [instruction]
        task = model.create_tasks(texts=language_instruction)

        episode_return = 0.0
        t = 0
        while t < max_episode_len + cfg.settle_time:
            if t < cfg.settle_time:
                obs, reward, done, trunc, _ = env.step([[0, 0, 0, 0, 0, 0, -1]] * 4)
                t += 1
                continue

            actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
            action = np.array(actions[0])
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)

            obs, reward, done, trunc, _ = env.step(action)
            # episode_return += reward
            episode_return = max(episode_return, reward)
            if done or trunc:
                break
            t += 1
        logging.info(f"Episode {i} return: {episode_return}")
        episode_returns.append(episode_return)

    env.close()
    return episode_returns, num_subtasks


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    test_case_files = [cfg.test_cases_config_1, cfg.test_cases_config_2]  # , cfg.test_cases_config_3]
    results = {}
    output_dir = to_absolute_path(cfg.output_dir)

    # test_cases = []

    # Add evaluation on all tasks in libero_90 suite except those starting with STUDY_SCENE
    # benchmark_dict = benchmark.get_benchmark_dict()
    # libero_goal_tasks = benchmark_dict["libero_goal"]().tasks
    # for task in libero_goal_tasks:
    #     # if not task.name.startswith("STUDY_SCENE"):
    #     test_cases.append(
    #         {
    #             "task_name": task.name,
    #             "instruction": task.language.capitalize() + ".",  # Assuming each task has an instruction attribute
    #             "task_suite": "libero_goal",
    #         }
    #     )
    for test_cases_config in test_case_files:
        test_cases = load_test_cases(to_absolute_path(f"evaluation_pipeline/{test_cases_config}"))
        test_cases_name = Path(test_cases_config).stem.split(".")[0]

        for checkpoint in [
            "subtasks_cumul_aug",
        ]:  # "baseline", "subtasks_augmented" "subtasks_mix_2", "original_mix"
            if checkpoint == "baseline":
                model_path = "hf://rail-berkeley/octo-small-1.5"
            else:
                model_path = os.path.join(cfg.finetuned_path, checkpoint)
            logging.info(f"Loading model from {model_path}...")
            if checkpoint == "baseline":
                model = OctoModel.load_pretrained(model_path)
            else:
                model = OctoModel.load_pretrained(model_path, cfg.model_step)

            # Load existing results if they exist
            results_file = os.path.join(output_dir, f"{test_cases_name}_{checkpoint}_results.json")
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    existing_results = json.load(f)
                logging.info(f"Loaded existing results from {results_file}")
            else:
                existing_results = []

            results[checkpoint] = existing_results
            existing_tasks = {(res["task_name"], res["instruction"], res["task_suite"]) for res in existing_results}

            for case in test_cases:
                task_name = case["task_name"]
                instruction = case["instruction"]
                task_suite = case["task_suite"]
                related_task = case.get("related")

                # Skip if the task has already been evaluated
                if (task_name, instruction, task_suite) in existing_tasks:
                    logging.info(f"Skipping already evaluated task: {task_name}, {instruction}, {task_suite}")
                    continue

                episode_returns, num_subtasks = evaluate_task(
                    cfg, model, task_name, task_suite, instruction, related_task
                )
                result = {
                    "instruction": instruction,
                    "task_name": task_name,
                    "task_suite": task_suite,
                    "success_rates": episode_returns,
                    "num_subtasks": num_subtasks,
                }
                results[checkpoint].append(result)
                logging.info(
                    f"Checkpoint: {checkpoint} Task: {task_name} Instruction: {instruction} Success Rate: {np.mean(episode_returns)}"
                )
                save_results(results[checkpoint], f"{test_cases_name}_{checkpoint}_results", output_dir)


if __name__ == "__main__":
    main()
