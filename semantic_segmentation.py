from google.genai import types
from google.genai.errors import ServerError, ClientError
import cv2

import os
import re
import time
import json
from pathlib import Path
import argparse

from utils import part_from_file, get_task_instruction_libero, init_genai_client, SemanticBreakdownGenerator


model_name = "gemini-2.5-pro-exp-03-25"
# model_name = "gemini-2.0-flash"
client = init_genai_client()


def get_first_frame(video_path):
    vidcap = cv2.VideoCapture(str(video_path))
    success, image = vidcap.read()
    if success:
        name = str(video_path.with_name(f"{video_path.stem}_first_frame.png"))
        cv2.imwrite(name, image)
        return name
    else:
        raise ValueError(f"Failed to read video file {video_path}.")


def create_prompt(task_instruction, frame):
    base_prompt = f"""You are a robot arm with a simple gripper. You are given the task "{task_instruction}" Using the task and the image showing the layout of the scene, create a plan of the possible sub-task motions you will need to perform to complete the task."""

    query_prompt = (
        base_prompt
        + f"""\n\nFollow these guidelines:
1. Start off by pointing to no more than 8 items in the image."""
        + """ The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]."""
        + f""" The points are in [y, x] format normalized to 0-1000. One element a line.
2. Think step by step and list the locations and relations of all objects, noting any object that might interfere with the task.
3. Finally, output the sub-tasks that will need to be completed to "{task_instruction}" in this scene. Try to be concise."""
        + """ Output in the json format:  [{"sub_task": <sub-task>, "description": <description>}, ...]

Example sub-tasks: "Grasp the bowl to the right of the white mug", "Move the mug towards the red plate", "Rotate the knob", "Pull the top drawer open", "Push the bottom drawer closed", "Place the can on top of the cabinet", "Lift the middle bowl"
    """
    )

    return [types.Content(role="user", parts=[frame, types.Part(text=query_prompt)])]


def extract_subtask_labels(frame_part, task_instruction):
    messages = []

    messages = create_prompt(task_instruction, frame_part, zero_shot=True)

    response = client.models.generate_content(
        model=model_name,
        contents=messages,
        config=types.GenerateContentConfig(temperature=0.0),
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
        with open(labels_json_out_path, "r") as f:
            labels_json_dict = json.load(f)

    metadata_path = Path(args.input_dir) / "metadata.json"
    if not metadata_path.exists():
        print(f"Metadata file not found at {metadata_path}")
        metadata = None
    else:
        with open(metadata_path) as f:
            metadata = json.load(f)

    seen_instructions = set()
    video_tasks = []
    for mp4_file in Path(args.input_dir).glob("*.mp4"):
        traj_filename = mp4_file.stem.split("_demo")[0] + "_demo"

        if metadata:
            file_metadata = metadata.get(mp4_file.name)
            scene = file_metadata["scene"]
            task_instruction = file_metadata["task"]
        else:
            scene, task_instruction, _ = get_task_instruction_libero(mp4_file)
        grounded_instruction = scene + "_" + task_instruction
        if grounded_instruction in seen_instructions:
            print(f"Skipping duplicate task instruction: {task_instruction} in scene {scene}")
            continue
        seen_instructions.add(grounded_instruction)

        if traj_filename in labels_json_dict:
            print(f"Skipping already processed file: {mp4_file.name}")
            continue

        video_tasks.append((mp4_file, traj_filename, scene, task_instruction))

    zero_shot = True
    for mp4_file, traj_filename, scene, task_instruction in video_tasks:
        print(f"Processing file: {mp4_file.name}")
        time.sleep(5)  # Avoid rate limiting
        first_frame = get_first_frame(mp4_file)
        first_frame_part = part_from_file(client, first_frame)
        os.remove(first_frame)
        try:
            response = extract_subtask_labels(first_frame_part, task_instruction)
        except ServerError as e:
            print(f"Error processing file {mp4_file.name}: {e}")
            print("Retrying...")
            time.sleep(10)
            try:
                response = extract_subtask_labels(first_frame_part, task_instruction)
            except Exception as e:
                print(f"Error processing file {mp4_file.name} again: {e}")
                continue
        except ClientError as e:
            print(f"Client error: {e}")
            break

        if zero_shot:
            json_blocks = re.findall(r"```json\s*(.*?)\s*```", response.text, re.DOTALL)
            subtask_list = json.loads(json_blocks[1])
            subtask_labels = ""
            for i, subtask in enumerate(subtask_list):
                subtask_labels += f"{i+1}. **{subtask['sub_task']}:** {subtask['description']}\n"
        else:
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
    parser = argparse.ArgumentParser(description="Extract semantic subtasks from video files.")
    parser.add_argument("--input_dir", type=Path, help="Path to the input dir of video files", required=True)
    parser.add_argument("--output_dir", type=Path, help="Directory to save output labels", required=True)
    parser.add_argument("--batched_inference", action="store_true", help="Use batched inference for multiple videos")
    args = parser.parse_args()

    main(args)
