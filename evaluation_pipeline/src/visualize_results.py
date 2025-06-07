import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


test_case_1_idx_map = {
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": 0,
    "KITCHEN_SCENE2_open_the_bottom_drawer_of_the_cabinet": 5,
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": 9,
    "KITCHEN_SCENE10_put_the_black_bowl_on_top_of_the_cabinet": 7,
    "LIVING_ROOM_SCENE2_pick_up_the_cream_cheese_box_and_put_it_in_the_basket": 8,
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it": 3,
    "KITCHEN_SCENE4_open_the_top_drawer_of_the_cabinet": 6,
}

test_case_2_idx_map = {
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": 0,
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": 1,
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": 2,
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": 3,
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": 4,
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": 5,
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": 6,
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet": 7,
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet": 8,
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet": 9,
    "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it": 10,
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it": 11,
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it": 12,
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer": 13,
}

test_case_3_idx_map = {
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": 3,
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": 9,
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": 8,
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": 4,
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": 7,
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": 1,
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": 6,
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": 2,
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": 0
}


def visualize_checkpoint_comparison(checkpoint_data_dict, test_case_name, output_dir, baseline_data=None):
    # Define colors using a more professional color palette
    colors = {
        "baseline": "#4B4B4B",  # Dark gray
        "original_mix": "#FF6B6B",  # Coral red
        "subtasks_cumul_aug": "#4ECDC4",  # Turquoise
        "subtasks_mix_2": "#45B7D1",  # Ocean blue
        "subtasks_augmented": "#96CEB4",  # Sage green
    }

    legend_labels = {
        "baseline": r"$\pi_0$",  # Octo
        "original_mix": r"$\pi_0$ - Finetuned",  # "Octo - Finetuned",
        "subtasks_mix_2": "TREAD w/o diverse labels",
        "subtasks_augmented": "TREAD",
        "subtasks_cumul_aug": "TREAD with grouping",
    }

    title = {
        "test_cases_1": "Motion Generalization",
        "test_cases_2": "Language Generalization",
        "test_cases_3": "LIBERO-10",
    }

    # Set global font sizes and style
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["axes.titlesize"] = 22
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 20
    plt.style.use("seaborn-v0_8-darkgrid")

    # Separate single, two and three step tasks for all checkpoints
    checkpoint_tasks = {
        checkpoint: {
            "single": [r for r in data if r["num_subtasks"] == 1],
            "two": [r for r in data if r["num_subtasks"] == 2],
            "three": [r for r in data if r["num_subtasks"] == 3],
        }
        for checkpoint, data in checkpoint_data_dict.items()
    }

    # For single step tasks
    if any(len(tasks["single"]) > 0 for tasks in checkpoint_tasks.values()):
        plt.figure(figsize=(12, 7))

        success_rates = []
        error_bars = []
        baseline_rates = []
        baseline_errors = []
        labels = []
        color_list = []

        for checkpoint, tasks in checkpoint_tasks.items():
            if tasks["single"]:
                # Calculate mean and standard error for test case
                task_means = [np.mean(task["success_rates"]) for task in tasks["single"]]
                rate = np.mean(task_means)
                error = np.std(task_means) / np.sqrt(len(task_means))

                # Calculate baseline performance if available
                if baseline_data and checkpoint in baseline_data:
                    baseline_means = [
                        np.mean(task["success_rates"])
                        for task in baseline_data[checkpoint]
                        if task["num_subtasks"] == 1
                    ]
                    baseline_rate = np.mean(baseline_means)
                    baseline_error = np.std(baseline_means) / np.sqrt(len(baseline_means))
                else:
                    baseline_rate = None
                    baseline_error = None

                success_rates.append(rate)
                error_bars.append(error)
                baseline_rates.append(baseline_rate)
                baseline_errors.append(baseline_error)
                labels.append(legend_labels[checkpoint])
                color_list.append(colors[checkpoint])

        x = np.arange(len(labels))
        width = 0.35

        # Plot baseline bars first (if available)
        # if any(r is not None for r in baseline_rates):
        #     bars2 = plt.bar(
        #         x,
        #         baseline_rates,
        #         width,
        #         label="Reference",
        #         color=color_list,
        #         alpha=0.5,
        #         edgecolor="black",
        #         linewidth=1.5,
        #         # yerr=baseline_errors,
        #         capsize=10,
        #         error_kw={"elinewidth": 2, "capthick": 2},
        #     )

        # Plot test case bars on top
        bars1 = plt.bar(
            x,
            success_rates,
            width,
            label="Performance",
            color=color_list,
            edgecolor="black",
            linewidth=1.5,
            yerr=error_bars,
            capsize=10,
            error_kw={"elinewidth": 2, "capthick": 2},
        )

        # Add value labels for both bars
        for i, (perf, ref) in enumerate(zip(success_rates, baseline_rates)):
            if ref is not None:
                plt.text(
                    x[i],
                    ref + baseline_errors[i] + 0.02,
                    f"{ref:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    alpha=0.5,
                )
            plt.text(x[i], perf + error_bars[i] + 0.02, f"{perf:.2f}", ha="center", va="bottom", fontsize=12, alpha=0.7)

        plt.ylabel("Success Rate")
        plt.title(title[test_case_name] + "\nSingle Goal Tasks")
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylim(0, 1)
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{test_case_name}_single_step_comparison.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    # For multi-step tasks
    if any(len(tasks["two"]) > 0 for tasks in checkpoint_tasks.values()):
        plt.figure(figsize=(14, 8))

        def get_step_proportions_with_error(tasks, num_steps):
            if not tasks:
                return [0] * (num_steps + 1), [0] * (num_steps + 1)

            # Calculate per-task success rates
            task_rates = []
            for task in tasks:
                rates = np.zeros(num_steps + 1)
                success_counts = np.array([int(rate * num_steps * 2) // 2 for rate in task["success_rates"]])
                for i in range(num_steps + 1):
                    rates[i] = np.mean(success_counts >= i)
                task_rates.append(rates)

            task_rates = np.array(task_rates)
            means = np.mean(task_rates, axis=0)
            errors = np.std(task_rates, axis=0) / np.sqrt(len(task_rates))  # Standard error
            return means, errors

        # Get proportions and errors for each checkpoint
        checkpoint_props = {}
        checkpoint_errors = {}
        baseline_props = {}
        baseline_errors = {}

        for checkpoint, tasks in checkpoint_tasks.items():
            if tasks["two"]:
                props, errors = get_step_proportions_with_error(tasks["two"], 2)
                checkpoint_props[checkpoint] = props[1:]
                checkpoint_errors[checkpoint] = errors[1:]

                # Get baseline data if available
                if baseline_data and checkpoint in baseline_data:
                    baseline_tasks = [t for t in baseline_data[checkpoint] if t["num_subtasks"] == 2]
                    if baseline_tasks:
                        props, errors = get_step_proportions_with_error(baseline_tasks, 2)
                        baseline_props[checkpoint] = props[1:]
                        baseline_errors[checkpoint] = errors[1:]

        x = np.arange(2)
        width = 0.15  # Narrower bars to fit pairs

        # Plot test case bars
        for i, (checkpoint, props) in enumerate(checkpoint_props.items()):
            offset = width * (i - (len(checkpoint_props) - 1) / 2)
            bars1 = plt.bar(
                x + offset - width / 2,
                props,
                width,
                label=legend_labels[checkpoint],
                color=colors[checkpoint],
                edgecolor="black",
                linewidth=1.5,
                yerr=checkpoint_errors[checkpoint],
                capsize=10,
                error_kw={"elinewidth": 2, "capthick": 2},
            )

            # Add value labels on test case bars
            for j, v in enumerate(props):
                plt.text(
                    x[j] + offset - width / 2,
                    v + checkpoint_errors[checkpoint][j] + 0.02,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )

            # Plot baseline bars if available
            # if checkpoint in baseline_props:
            #     bars2 = plt.bar(
            #         x + offset - width / 2,
            #         baseline_props[checkpoint],
            #         width,
            #         # label=f"{legend_labels[checkpoint]} (Ref)" if i == 0 else "",
            #         color=colors[checkpoint],
            #         alpha=0.5,
            #         edgecolor="black",
            #         linewidth=1.5,
            #         # yerr=baseline_errors[checkpoint],
            #         capsize=10,
            #         error_kw={"elinewidth": 2, "capthick": 2},
            #     )

            # Add value labels on baseline bars
            # for j, v in enumerate(baseline_props[checkpoint]):
            #     plt.text(
            #         x[j] + offset - width / 2,
            #         v + baseline_errors[checkpoint][j] + 0.02,
            #         f"{v:.2f}",
            #         ha="center",
            #         va="bottom",
            #         fontsize=12,
            #     )

        plt.ylabel("Success Rate")
        plt.title(title[test_case_name] + "\nTwo Goal Tasks")
        plt.xticks(x, [str(i + 1) for i in range(2)])
        plt.xlabel("Number of Goals Completed")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{test_case_name}_two_step_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()


def process_pi0_format_results(results_data, task_subtasks):
    """Convert the new format results into the old format structure."""
    processed_results = []
    task_subtasks_idxs = {f"libero_10_{test_case_3_idx_map[task]}": v for task, v in task_subtasks.items()}

    # Group results by task
    task_results = {}

    # Process successes (1.0)
    for run in results_data.get("run_successes", []):
        task = run.split("_init_state_")[0]
        if task not in task_results:
            task_results[task] = {"success_rates": []}
        task_results[task]["success_rates"].append(1.0)

    # Process partial successes (0.5)
    for run in results_data.get("partial_successes", []):
        task = run.split("_init_state_")[0]
        if task not in task_results:
            task_results[task] = {"success_rates": []}
        task_results[task]["success_rates"].append(0.5)

    # Process failures (0.0)
    for run in results_data.get("run_failures", []):
        task = run.split("_init_state_")[0]
        if task not in task_results:
            task_results[task] = {"success_rates": []}
        task_results[task]["success_rates"].append(0.0)

    # Convert to list format and add correct number of subtasks
    for task, data in task_results.items():
        task = task.split("_init_state_")[0]  # Remove init_state from task name

        if task in task_subtasks_idxs:  # task_subtasks:
            data["num_subtasks"] = task_subtasks_idxs[task]  # task_subtasks[task]
        else:
            print(f"Warning: Task {task} not found in test cases definition")
            data["num_subtasks"] = 1  # Default fallback
        processed_results.append(data)

    return processed_results


def load_test_cases(test_cases_file):
    """Load test cases and create a mapping of task names to their number of subtasks."""
    with open(test_cases_file, "r") as f:
        test_cases = json.load(f)

    # Create mapping of task name to number of subtasks
    task_subtasks = {}
    for task in test_cases:
        task_name = task["task_name"]
        task_subtasks[task_name] = task["num_subtasks"]

    return task_subtasks


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    test_case_files = [cfg.test_cases_config_3]
    output_dir = to_absolute_path(cfg.output_dir)
    use_new_format = cfg.get("use_new_format", False)

    for test_cases_config in test_case_files:
        test_cases_config = to_absolute_path(test_cases_config)
        test_cases_name = Path(test_cases_config).stem.split(".")[0]

        # Load test cases to get number of subtasks
        task_subtasks = load_test_cases(test_cases_config)

        # Load results for all checkpoints
        checkpoints = [
            "original_mix",
            "subtasks_mix_2",
            "subtasks_augmented",
        ]
        checkpoint_data = {}
        baseline_data = {}

        # Load baseline data if it exists
        baseline_config = test_cases_name + "_baseline"
        for checkpoint in checkpoints:
            # Load regular results
            if use_new_format:
                results_file = os.path.join(output_dir, "pi0", test_cases_name, f"{checkpoint}.json")
            else:
                results_file = os.path.join(output_dir, f"{test_cases_name}_{checkpoint}_results.json")
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    data = json.load(f)
                    if use_new_format:
                        checkpoint_data[checkpoint] = process_pi0_format_results(data, task_subtasks)
                    else:
                        checkpoint_data[checkpoint] = data

            # Load baseline results
            # baseline_file = os.path.join(output_dir, f"{baseline_config}_{checkpoint}_results.json")
            # if os.path.exists(baseline_file):
            #     with open(baseline_file, "r") as f:
            #         data = json.load(f)
            #         if use_new_format:
            #             baseline_data[checkpoint] = process_pi0_format_results(data)
            #         else:
            #             baseline_data[checkpoint] = data

        if checkpoint_data:  # Only create visualization if we have data
            visualize_checkpoint_comparison(checkpoint_data, test_cases_name, output_dir, baseline_data=baseline_data)


if __name__ == "__main__":
    main()
