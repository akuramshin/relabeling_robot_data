from google.genai import types
import cv2

import os
import json
from pathlib import Path
import argparse

from utils import upload_file, client, model_name


def get_first_frame(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    if success:
        name = str(video_path.with_name(f"{video_path.stem}_first_frame.png"))
        cv2.imwrite(name, image)
        return name
    else:
        raise ValueError(f"Failed to read video file {video_path}.")


def get_task_instruction(video_path):
    """The task instruction is assumed to be the name of the video file without the suffix _demo_{i}.mp4
    For example, we return "Move spoon." for "move_spoon_demo_1.mp4"
    """
    base_name = video_path.stem 
    task = base_name.split("_demo_")[0]
    task = task.replace("_", " ")
    task = task.capitalize()
    return task + "."


def extract_subtask_labels(video_path, task_instruction, output_dir):
    # video_file = upload_video(video_path)

    messages = []
    query_prompt = f"""You are a robot arm with a simple gripper. You are given the task "{task_instruction}" Given the task and the image showing the layout of the scene, create a plan of the possible sub-task motions you will need to perform to complete the task.

Review the example for the desired format.

Here is a plan outlining the sub-task motions to move the spoon to the right:

1. Grasp Spoon: Approach the spoon and use the gripper to grasp the spoon by the handle.
2. Lift Spoon: Move the gripper vertically upwards a small distance, just enough to lift the spoon clear of the table surface.
3. Move Right: Move the gripper horizontally to the right, carrying the spoon over the table surface to the desired new position.
4. Release Spoon: Move the gripper vertically downwards to place the spoon gently onto the table surface at the new location and open the gripper to release the spoon."""

    first_frame = get_first_frame(video_path)
    first_frame_part = upload_file(first_frame)
    messages.append(types.Content(role="user", parts=[first_frame_part, types.Part(text=query_prompt)]))

    response = client.models.generate_content(
        model=model_name,
        contents=messages,
    )

    return response


def main(args):
    print(f"VLM Processing {args.input_dir} dataset!")

    # Create target directory
    if os.path.isdir(args.output_dir):
        user_input = input(
            f"Target directory already exists at path: {args.output_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: "
        )
        if user_input != "y":
            exit()
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare JSON file to record VLM api responses
    labels_json_dict = {}
    labels_json_out_path = f"{args.output_dir}/dataset_subtask_labels.json"
    with open(labels_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(labels_json_dict, f)

    # Process all mp4 files in the dataset directory
    for mp4_file in Path(args.input_dir).glob("*.mp4"):
        print(f"Processing file: {mp4_file}")
        task_instruction = get_task_instruction(mp4_file)
        response = extract_subtask_labels(mp4_file, task_instruction, args.output_dir)

        # Save the response to the JSON file
        subtask_labels = response.text.split(":", 1)[1].strip()
        labels_json_dict[mp4_file.stem] = {"task_instruction": task_instruction, "subtask_labels": subtask_labels}
        print(f"Saved response for {mp4_file.stem} to {labels_json_out_path}")
        print(f"Semantic sub-task labels: {response.text}")
        print("---------------------------------------------------")

    with open(labels_json_out_path, "w") as f:
        json.dump(labels_json_dict, f, indent=4)
    print(f"Saved all results to {labels_json_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract videos from HDF5 dataset files.")
    parser.add_argument("--input_dir", type=Path, help="Path to the input dir of video files", required=True)
    parser.add_argument("--output_dir", type=Path, help="Directory to save output labels", required=True)
    args = parser.parse_args()

    main(args)
