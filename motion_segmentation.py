from google.genai import types
from google.genai.errors import ServerError, ClientError

import os
import time
import json
from pathlib import Path
import argparse

from utils import upload_file, client, model_name, get_task_instruction


example_files = [
    upload_file(file_name="./examples/close_the_top_drawer.mp4"),
    upload_file(file_name="./examples/open_the_microwave.mp4"),
    upload_file(file_name="./examples/put_mug_on_plate.mp4"),
]
example_user_message = [
    types.Part.from_text(
        text="""The video is showing a robot arm performing the task \"Close the top drawer of the cabinet.\" The task can be broken down into these possible sub-task motions:

1.  **Grasp Handle:** Move the gripper to the handle of the *top* drawer. Position the gripper around the handle and close it firmly.
2.  **Push Drawer Closed:** Move the gripper horizontally inwards, towards the cabinet face, pushing the drawer until it is fully closed. Maintain a firm grip while pushing.

Carefully look at the video and describe the sub-task motions you see being performed by the robot and the corresponding timecodes."""
    ),
    types.Part.from_text(
        text="""The video is showing a robot arm performing the task \"Open the microwave.\" The task can be broken down into these possible sub-task motions:

1.  **Approach Handle:** Move the gripper to the handle located on the left side of the microwave door.
2.  **Pull Door Open:** Move the gripper horizontally outwards (towards the left, away from the microwave body), pulling the door open.

Carefully look at the video and describe the sub-task motions you see being performed by the robot and the corresponding timecodes."""
    ),
    types.Part.from_text(
        text="""The video is showing a robot arm performing the task \"Put the white mug on the left plate.\" The task can be broken down into these possible sub-task motions:

1.  **Grasp Mug:** Approach the white mug (the one on the right side of the image). Position the gripper around the body or handle of the mug and close the gripper to grasp it securely.
2.  **Lift Mug:** Move the gripper vertically upwards, lifting the mug clear of the table surface and the other mugs.
3.  **Move Mug to Plate:** Move the gripper horizontally, carrying the mug across the table until it is positioned directly above the plate on the *left*.
4.  **Place Mug on Plate:** Move the gripper vertically downwards to gently place the white mug onto the surface of the left plate. Open the gripper to release the mug.

Carefully look at the video and describe the sub-task motions you see being performed by the robot and the corresponding timecodes."""
    ),
]
example_model_message = [
    types.Part.from_text(
        text="""Okay, here's a breakdown of the sub-task motions shown in the video:

1.  **00:00- 00:04 Move gripper to front of drawer:** The robot arm moves downwards and towards the open top drawer, positioning its open gripper near the handle, against the front face of the drawer. It does *not* grasp the handle itself.
2.  **00:04 - 00:07 Push drawer closed:** The robot arm moves horizontally inwards, pushing the front face of the drawer until it is fully closed."""
    ),
    types.Part.from_text(
        text="""Okay, here's the breakdown of the sub-task motions for opening the microwave:

1.  **00:00 - 00:04 Approach Handle:** The robot arm moves downwards and towards the microwave handle on the left side of the door.
2.  **00:04 - 00:07 Pull Door Open:** The robot arm moves horizontally outwards (to the left), pushing against the handle/door edge to swing the microwave door open."""
    ),
    types.Part.from_text(
        text="""Okay, here's the breakdown of the sub-task motions for placing the white mug on the left plate:

1.  **00:00 - 00:03 Grasp Mug:** The robot moves its gripper downwards and positions it around the white mug (on the right), then closes the gripper to grasp it.
2.  **00:03 - 00:04 Lift Mug:** The robot lifts the grasped mug vertically upwards, clear of the table.
3.  **00:04 - 00:06 Move Mug to Plate:** The robot moves the mug horizontally towards the left, positioning it above the left plate.
4.  **00:06 - 00:08 Place Mug on Plate:** The robot lowers the mug down onto the surface of the left plate and then opens its gripper to release the mug."""
    ),
]


def create_prompt(task_instruction, video, subtask_labels, zero_shot=False):
    query_prompt = f"""The video is showing a robot arm performing the task \"{task_instruction}\" The task can be broken down into these possible sub-task motions:

{subtask_labels}

Carefully look at the video and describe the sub-task motions you see being performed by the robot and the corresponding timecodes."""

    if zero_shot:
        return [types.Content(role="user", parts=[video, types.Part(text=query_prompt)])]

    contents = [
        types.Content(role="user", parts=[example_files[0], example_user_message[0]]),
        types.Content(role="model", parts=[example_model_message[0]]),
        types.Content(role="user", parts=[example_files[1], example_user_message[1]]),
        types.Content(role="model", parts=[example_model_message[1]]),
        types.Content(role="user", parts=[example_files[2], example_user_message[2]]),
        types.Content(role="model", parts=[example_model_message[2]]),
        types.Content(role="user", parts=[video, types.Part.from_text(text=query_prompt)]),
    ]
    return contents


def extract_subtask_labels(video_part, task_instruction, subtask_labels):
    messages = []

    messages = create_prompt(task_instruction, video_part, subtask_labels)

    response = client.models.generate_content(
        model=model_name,
        contents=messages,
    )

    return response


def main(args):
    print(f"VLM Processing {args.input_dir} dataset!")

    os.makedirs(args.output_dir, exist_ok=True)

    labels_json_dict = {}
    labels_json_out_path = f"{args.output_dir}/dataset_motion_labels.json"

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

    # Load previous labels if available
    if os.path.exists(args.subtask_labels):
        with open(args.subtask_labels, "r") as f:
            subtask_labels_dict = json.load(f)

    for mp4_file in Path(args.input_dir).glob("*.mp4"):
        if not mp4_file.exists():
            print(f"Warning: Video file {mp4_file} not found, skipping")
            continue

        scene_task_id = mp4_file.stem.split("_demo")[0] + "_demo"
        if metadata:
            file_metadata = metadata.get(mp4_file.name)
            task_instruction = file_metadata["task"]
            demo_id = file_metadata["demo_id"]
        else:
            _, task_instruction, demo_id = get_task_instruction(mp4_file)
        if demo_id == -1:
            print(f"Warning: No demo ID found for {mp4_file.name}, skipping")
            continue
        traj_id = scene_task_id + f"_{demo_id}"

        if traj_id in labels_json_dict:
            print(f"Skipping already processed file: {mp4_file.name}")
            continue

        print(f"Processing file: {mp4_file.name}")
        if scene_task_id not in subtask_labels_dict:
            print(f"Warning: No subtask labels found for {scene_task_id}, skipping")
            continue
        subtask_labels = subtask_labels_dict.get(scene_task_id).get("subtask_labels")
        video_part = upload_file(mp4_file)
        time.sleep(5)  # Avoid rate limiting
        try:
            response = extract_subtask_labels(video_part, task_instruction, subtask_labels)
        except ServerError as e:
            print(f"Error processing file {mp4_file.name}: {e}")
            print("Retrying...")
            time.sleep(10)
            try:
                response = extract_subtask_labels(video_part, task_instruction, subtask_labels)
            except Exception as e:
                print(f"Error processing file {mp4_file.name} again: {e}")
                continue
        except ClientError as e:
            print(f"Client error: {e}")
            break

        response_lines = response.text.split(":", 1)[1].strip().split("\n")
        subtask_labels = [line for line in response_lines if line.strip() and line.strip()[0].isdigit()]
        labels_json_dict[traj_id] = {
            "demo_id": demo_id,
            "motion_labels": subtask_labels,
            "response": response.text,
        }
        print(f"Motion sub-task timecodes: {response.text}")
        print("---------------------------------------------------")

        with open(labels_json_out_path, "w") as f:
            json.dump(labels_json_dict, f, indent=4)

    print(f"Saved all results to {labels_json_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract motion subtasks from video files.")
    parser.add_argument("--input_dir", type=Path, help="Path to the input dir of video files", required=True)
    parser.add_argument("--subtask_labels", type=Path, help="Path to the subtask labels json file", required=True)
    parser.add_argument("--output_dir", type=Path, help="Directory to save output labels", required=True)
    args = parser.parse_args()

    main(args)
