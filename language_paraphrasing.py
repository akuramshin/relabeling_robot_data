from google.genai import types
from google.genai.errors import ServerError, ClientError
import cv2

import os
import re
import time
import json
from pathlib import Path
import argparse

from utils import upload_file, part_from_file, get_task_instruction_libero, init_genai_client, ParaphraseGenerator

model_name = "gemini-2.0-flash-001"
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


def create_prompt(task_instruction, frame, subtask_labels):
    query_prompt = (
        f"""You are a robot arm with a simple gripper. You were given the task "{task_instruction}" Here is a break down of the sub-task motions you need to perform to complete this task.

{subtask_labels}

Help yourself understand the goal by writing 5 alternate descriptions. For the overall task, and each of the sub-task motions write 5 other ways to instruct the same motion.

Follow these guidelines:
1. Start off by pointing to no more than 8 items in the image."""
        + """ The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000. One element a line.
2. Then describe each object both in terms of their properties and their spatial relation to each other and in the global scene (e.g. "the smaller object", "the black object", "the middle object" ).
3. For the final step rewrite the original task and subtasks using your object descriptions in the json format:  [{"task": [<paraphrase-task>,], "subtasks": [[<paraphrase-sub-task>,], [<paraphrase-sub-task>,], ]}, ...]
"""
    )

    return [types.Content(role="user", parts=[frame, types.Part(text=query_prompt)])]


def generate_paraphrases(frame_part, task_instruction, subtask_labels):
    messages = []

    messages = create_prompt(task_instruction, frame_part, subtask_labels)

    response = client.models.generate_content(
        model=model_name,
        contents=messages,
        config=types.GenerateContentConfig(temperature=1.0),
    )

    return response


def main(args):
    print(f"VLM Processing {args.input_dir} dataset!")

    os.makedirs(args.output_dir, exist_ok=True)

    labels_json_dict = {}
    labels_json_out_path = f"{args.output_dir}/dataset_paraphrases_labels.json"

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
    
    with open(args.subtask_labels, "r") as f:
        subtask_labels_dict = json.load(f)

    video_tasks = []
    for mp4_file in Path(args.input_dir).glob("*.mp4"):
        scene_task_id = mp4_file.stem.split("_demo")[0] + "_demo"
        if metadata:
            file_metadata = metadata.get(mp4_file.name)
            demo_id = file_metadata["demo_id"]
            task_instruction = file_metadata["task"]
        else:
            _, task_instruction, demo_id = get_task_instruction_libero(mp4_file)
        
        traj_id = scene_task_id + f"_{demo_id}"

        if traj_id in labels_json_dict:
            print(f"Skipping already processed file: {mp4_file.name}")
            continue

        if traj_id not in subtask_labels_dict:
            print(f"Warning: No motion subtask labels found for {traj_id}, skipping")
            continue

        subtask_list = subtask_labels_dict.get(traj_id).get("motion_labels")
        subtask_labels = ""
        for i, subtask in enumerate(subtask_list):
            subtask_labels += f"{i+1}. {subtask['sub_task']}\n"

        video_tasks.append((mp4_file, traj_id, task_instruction, subtask_labels))

    if args.batched_inference:
        print("Running in batched inference mode.")
        # Prepare batch inputs
        batch_inputs = []
        t = 0
        for mp4_file, traj_id, task_instruction, subtask_labels in video_tasks:
            if t >= 2:
                break
            first_frame = get_first_frame(mp4_file)
            first_frame_part = upload_file(client, first_frame)
            os.remove(first_frame)
            batch_inputs.append(
                {
                    "traj_id": traj_id,
                    "task_instruction": task_instruction,
                    "image_url": first_frame_part.uri,
                    "subtask_labels": subtask_labels,
                }
            )
            t += 1

        subtask_dataset = ParaphraseGenerator(
            model_name=model_name,
            backend="gemini",
            batch=True,
            backend_params={"batch_size": 2},
        )
        # Run batched inference
        results = subtask_dataset(batch_inputs)
        # Save results
        for result in results:
            for traj_id, data in result.items():
                labels_json_dict[traj_id] = data
                print(f"Paraphrases: for {traj_id}: {data['subtask_labels']}")
                print("---------------------------------------------------")
        with open(labels_json_out_path, "w") as f:
            json.dump(labels_json_dict, f, indent=4)
    else:
        for mp4_file, traj_id, task_instruction, subtask_labels in video_tasks:
            print(f"Processing file: {mp4_file.name}")
            time.sleep(5)  # Avoid rate limiting
            first_frame = get_first_frame(mp4_file)
            first_frame_part = part_from_file(client, first_frame)
            os.remove(first_frame)
            try:
                response = generate_paraphrases(first_frame_part, task_instruction, subtask_labels)
            except ServerError as e:
                print(f"Error processing file {mp4_file.name}: {e}")
                print("Retrying...")
                time.sleep(10)
                try:
                    response = generate_paraphrases(first_frame_part, task_instruction, subtask_labels)
                except Exception as e:
                    print(f"Error processing file {mp4_file.name} again: {e}")
                    continue
            except ClientError as e:
                print(f"Client error: {e}")
                break

            json_blocks = re.findall(r"```json\s*(.*?)\s*```", response.text, re.DOTALL)
            try:
                paraphrases_list = json.loads(json_blocks[1])
            except (json.JSONDecodeError, IndexError) as e:
                print(f"Error parsing JSON from response: {e}")
                paraphrases_list = []

            labels_json_dict[traj_id] = {
                "paraphrases_list": paraphrases_list,
                "response": response.text,
            }
            print(f"Paraphrases: {paraphrases_list}")
            print("---------------------------------------------------")

            with open(labels_json_out_path, "w") as f:
                json.dump(labels_json_dict, f, indent=4)

    print(f"Saved all results to {labels_json_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate task and subtask paraphrases with visual grounding.")
    parser.add_argument("--input_dir", type=Path, help="Path to the input dir of video files", required=True)
    parser.add_argument("--subtask_labels", type=Path, help="Path to the motion subtask labels json file", required=True)
    parser.add_argument("--output_dir", type=Path, help="Directory to save output labels", required=True)
    parser.add_argument("--batched_inference", action="store_true", help="Use batched inference for multiple videos")
    args = parser.parse_args()

    main(args)
