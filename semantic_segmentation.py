from google.genai import types
from google.genai.errors import ServerError, ClientError
import cv2

import os
import time
import json
from pathlib import Path
import argparse
import re

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
    instruction = video_path.stem 
    pattern = r'_([a-z]+(?:_[a-z]+)*)_'

    # Find all matches in the string
    matches = re.findall(pattern, instruction)

    # Join the matches and format the result
    task = " ".join(matches).replace("_", " ").capitalize() + "."

    if "demo" in task.lower():
        task = task.split(" demo")[0] + "."

    return task


def extract_subtask_labels(video_path, task_instruction):
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
    os.remove(first_frame)
    messages.append(types.Content(role="user", parts=[first_frame_part, types.Part(text=query_prompt)]))

    response = client.models.generate_content(
        model=model_name,
        contents=messages,
    )

    return response


def main(args):
    print(f"VLM Processing {args.input_dir} dataset!")

    os.makedirs(args.output_dir, exist_ok=True)

    labels_json_dict = {}
    labels_json_out_path = f"{args.output_dir}/dataset_subtask_labels.json"

    # Check if output file exists and load previous results
    if os.path.exists(labels_json_out_path):
        print(f"Found existing labels file at {labels_json_out_path}, continuing from previous state")
        with open(labels_json_out_path, 'r') as f:
            labels_json_dict = json.load(f)

    metadata_path = Path(args.input_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
            
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Process files listed in metadata
    seen_instructions = set()
    for mp4_file in Path(args.input_dir).glob("*.mp4"):
        traj_filename = mp4_file.stem.split("_demo")[0]+"_demo"

        # Skip if this file was already processed
        if traj_filename in labels_json_dict:
            print(f"Skipping already processed file: {mp4_file}")
            continue

        file_metadata = metadata.get(traj_filename)
        if not mp4_file.exists():
            print(f"Warning: Video file {mp4_file} not found, skipping")
            continue

        task_instruction = file_metadata["task"]
        if task_instruction in seen_instructions:
            print(f"Skipping duplicate task instruction: {task_instruction}")
            continue
        seen_instructions.add(task_instruction)
        print(f"Processing file: {mp4_file}")
        task_instruction = get_task_instruction(mp4_file)
        time.sleep(5)  # Avoid rate limiting
        try:
            response = extract_subtask_labels(mp4_file, task_instruction)
        except ServerError as e:
            print(f"Error processing file {mp4_file}: {e}")
            print("Retrying...")
            time.sleep(10)
            try:
                response = extract_subtask_labels(mp4_file, task_instruction)
            except Exception as e:
                print(f"Error processing file {mp4_file} again: {e}")
                continue
        except ClientError as e:
            print(f"Client error: {e}")
            break

        subtask_labels = response.text.split(":", 1)[1].strip()
        labels_json_dict[traj_filename] = {
            "subtask_labels": subtask_labels,
            "response": response.text,
        }
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
