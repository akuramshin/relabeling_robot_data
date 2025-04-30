import json
import random
import argparse
from pathlib import Path

def main(args):
    with open(args.input_dir / "dataset_motion_labels.json", "r") as f:
        motion_labels = json.load(f)

    with open(args.input_dir / "dataset_paraphrases_labels.json", "r") as f:
        paraphrases_labels = json.load(f)

    augmented_motion_labels = {}

    task_relabel_count = 0
    subtask_relabel_count = 0
    for traj_id, motion_data in motion_labels.items():
        augmented_motion_labels[traj_id] = motion_data.copy()  # Copy existing data

        # Find corresponding paraphrases
        if traj_id not in paraphrases_labels:
            print(f"Warning: No paraphrases found for {traj_id}, skipping augmentation")
            continue

        paraphrases_data = paraphrases_labels[traj_id]
        
        # Relabel main instruction
        if random.random() < 0.25:
            if not paraphrases_data["paraphrases_list"]:
                raise ValueError(f"Paraphrases list is empty for {traj_id}")
            if "task" not in paraphrases_data["paraphrases_list"][0]:
                raise ValueError(f"Subtasks not found for {traj_id}")
            task_relabel_count += 1
            augmented_motion_labels[traj_id]["instruction"] = random.choice(paraphrases_data["paraphrases_list"][0]["task"])

        # Relabel subtasks
        if "motion_labels" in motion_data:
            augmented_subtasks = []
            for i, subtask in enumerate(motion_data["motion_labels"]):
                augmented_subtask = subtask.copy()
                if "sub_task" in subtask:
                    if "paraphrases_list" in paraphrases_data and paraphrases_data["paraphrases_list"]:
                        # Find corresponding subtask paraphrases
                        if len(paraphrases_data["paraphrases_list"]) > 0:
                            if random.random() < 0.5:
                                subtask_relabel_count += 1
                                if "subtasks" not in paraphrases_data["paraphrases_list"][0]:
                                    raise ValueError(f"Subtasks not found for {traj_id}")
                                augmented_subtask["sub_task"] = random.choice(paraphrases_data["paraphrases_list"][0]["subtasks"][i])
                augmented_subtasks.append(augmented_subtask)
            augmented_motion_labels[traj_id]["motion_labels"] = augmented_subtasks

    print(f"Relabeled {task_relabel_count} main instructions and {subtask_relabel_count} subtasks.")

    with open(args.input_dir / "dataset_motion_augmented_labels.json", "w") as f:
        json.dump(augmented_motion_labels, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment motion labels with paraphrases.")
    parser.add_argument("--input_dir", type=Path, help="Path to the input dir of labels", required=True)
    args = parser.parse_args()
    main(args)