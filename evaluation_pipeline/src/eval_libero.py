from functools import partial
import os
import json
import jax
import numpy as np
from absl import logging
import matplotlib.pyplot as plt
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


def evaluate_task(cfg: DictConfig, model, task_name, task_suite_name, instruction):
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
            unnormalization_statistics=dataset_statistics[dataset_name]["action"]#model.dataset_statistics[dataset_name]["action"],
        ),
    )

    if task_suite_name == "libero_90" or task_suite_name == "libero_10":
        initial_states = task_suite.get_task_init_states(task_id)

    # running rollouts
    episode_returns = []
    num_subtasks = len(env.env.env.env.parsed_problem["goal_state"])
    for i in tqdm(range(cfg.num_rollouts)):
        env.reset()

        if task_suite_name == "libero_90" or task_suite_name == "libero_10":
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


def visualize_checkpoint_comparison(checkpoint_data_dict, test_case_name, output_dir):
    # Define colors for each checkpoint
    colors = {
        'baseline': 'lightgray',
        'original_mix': 'lightcoral', 
        # 'subtasks_mix': 'lightgreen',
        'subtasks_mix_2': 'lightseagreen',
        'subtasks_augmented': 'lightskyblue'
    }
    legend_labels = {
        'baseline': 'Baseline',
        'original_mix': 'Original',
        # 'subtasks_mix': 'Subtasks Mix',
        'subtasks_mix_2': 'Decomposed',
        'subtasks_augmented': 'Contextual Diversity'
    }
    title = {
        "test_cases_1": "Test Cases 1",
        "test_cases_2": "Test Cases 2",
        "test_cases_3": "LIBERO-10",
    }
    
    # Separate single, two and three step tasks for all checkpoints
    checkpoint_tasks = {
        checkpoint: {
            'single': [r for r in data if r["num_subtasks"] == 1],
            'two': [r for r in data if r["num_subtasks"] == 2],
            'three': [r for r in data if r["num_subtasks"] == 3]
        }
        for checkpoint, data in checkpoint_data_dict.items()
    }
    
    # Plot single step tasks comparison
    if any(len(tasks['single']) > 0 for tasks in checkpoint_tasks.values()):
        plt.figure(figsize=(10, 6))
        
        # Calculate average success rates
        success_rates = []
        labels = []
        color_list = []
        for checkpoint, tasks in checkpoint_tasks.items():
            if tasks['single']:
                rate = np.mean([np.mean(task["success_rates"]) for task in tasks['single']])
                success_rates.append(rate)
                labels.append(legend_labels[checkpoint])
                color_list.append(colors[checkpoint])
        
        x = np.arange(len(labels))
        plt.bar(x, success_rates, width=0.6, color=color_list)
        
        plt.ylabel('Success Rate')
        plt.title(title[test_case_name] + ' - Single-Step Tasks')
        plt.xticks(x, labels, rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{test_case_name}_single_step_comparison.png'))
        plt.close()
    
    # Plot multi-step tasks comparison (2 and 3 steps)
    if any(len(tasks['two']) > 0 for tasks in checkpoint_tasks.values()):
        plt.figure(figsize=(12, 6))
        
        def get_step_proportions(tasks, num_steps):
            total_trials = sum(len(task["success_rates"]) for task in tasks)
            if total_trials == 0:
                return [0] * (num_steps + 1)
            
            counts = [0] * (num_steps + 1)  # [0 steps, 1 step, ..., n steps]
            for task in tasks:
                for rate in task["success_rates"]:
                    step_count = int(rate * num_steps * 2) // 2  # Convert rate to step count
                    counts[step_count] += 1
            return [count/total_trials for count in counts]
        
        # Get proportions for each checkpoint
        checkpoint_props = {}
        for checkpoint, tasks in checkpoint_tasks.items():
            if tasks['two']:
                checkpoint_props[checkpoint] = get_step_proportions(tasks['two'], 2)
            if tasks['three']:
                checkpoint_props[checkpoint] += np.array(get_step_proportions(tasks['three'], 3)[1:])
        
        x = np.arange(2)  # 0 to max steps completed
        width = 0.15  # Reduced width to accommodate more bars
        
        # Plot bars for each checkpoint
        for i, (checkpoint, props) in enumerate(checkpoint_props.items()):
            offset = width * (i - (len(checkpoint_props)-1)/2)
            plt.bar(x + offset, props[1:], width, label=legend_labels[checkpoint], 
                    color=colors[checkpoint])
        
        plt.ylabel('Proportion of Trials')
        plt.title(title[test_case_name] + " - Two Step Tasks")
        plt.xticks(x, [str(i) for i in range(2)])
        plt.xlabel('Number of Subtasks Completed')
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{test_case_name}_two_step_comparison.png'))
        plt.close()


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    test_case_files = [cfg.test_cases_config_1, cfg.test_cases_config_2, cfg.test_cases_config_3]
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
    if not cfg.visualize:
        for test_cases_config in test_case_files:
            test_cases = load_test_cases(to_absolute_path(f"evaluation_pipeline/conf/{test_cases_config}"))
            test_cases_name = test_cases_config.split('.')[0]

            for checkpoint in ["subtasks_grouped"]: #["subtasks_mix_2", "original_mix", "baseline"]:
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

                    # Skip if the task has already been evaluated
                    if (task_name, instruction, task_suite) in existing_tasks:
                        logging.info(f"Skipping already evaluated task: {task_name}, {instruction}, {task_suite}")
                        continue

                    episode_returns, num_subtasks = evaluate_task(cfg, model, task_name, task_suite, instruction)
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

    # Add visualization at the end
    if cfg.visualize:
        for test_cases_config in test_case_files:
            test_cases_name = test_cases_config.split('.')[0]
            
            # Load results for all checkpoints
            checkpoints = ["baseline", "original_mix", "subtasks_mix_2", "subtasks_augmented"]
            checkpoint_data = {}
            
            for checkpoint in checkpoints:
                results_file = os.path.join(output_dir, f"{test_cases_name}_{checkpoint}_results.json")
                if os.path.exists(results_file):
                    with open(results_file, "r") as f:
                        checkpoint_data[checkpoint] = json.load(f)
            
            if checkpoint_data:  # Only create visualization if we have data
                visualize_checkpoint_comparison(checkpoint_data, test_cases_name, output_dir)

if __name__ == "__main__":
    main()
