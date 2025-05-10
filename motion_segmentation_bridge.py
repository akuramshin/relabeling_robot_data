from google.genai import types
from google.genai.errors import ServerError, ClientError

import os
import re
import time
import json
from pathlib import Path
import argparse

from utils import part_from_file, get_task_instruction_bridge, init_genai_client, int_to_mmss

model_name = "gemini-2.5-pro-exp-03-25"
# model_name = "gemini-2.0-flash-001"
client = init_genai_client()

example_files = [
    part_from_file(client, file_name="./examples/close_the_top_drawer.mp4"),
    part_from_file(client, file_name="./examples/open_the_microwave.mp4"),
    part_from_file(client, file_name="./examples/put_mug_on_plate.mp4"),
]
example_user_message = [
    types.Part.from_text(
        text="""The video is showing a robot arm performing the task \"Close the top drawer of the cabinet.\" The task can be broken down into these possible sub-task motions:

1.  **Grasp the top drawer handle:** Move the gripper to the handle of the *top* drawer. Position the gripper around the handle and close it firmly.
2.  **Push the top drawer closed:** Move the gripper horizontally inwards, towards the cabinet face, pushing the drawer until it is fully closed. Maintain a firm grip while pushing.

Carefully look at the video and describe the sub-task motions you see being performed by the robot and the corresponding timecodes."""
    ),
    types.Part.from_text(
        text="""The video is showing a robot arm performing the task \"Open the microwave.\" The task can be broken down into these possible sub-task motions:

1.  **Approach the microwave door handle:** Move the gripper to the handle located on the left side of the microwave door.
2.  **Pull microwave door open:** Move the gripper horizontally outwards (towards the left, away from the microwave body), pulling the door open.

Carefully look at the video and describe the sub-task motions you see being performed by the robot and the corresponding timecodes."""
    ),
    types.Part.from_text(
        text="""The video is showing a robot arm performing the task \"Put the white mug on the right plate.\" The task can be broken down into these possible sub-task motions:

1.  **Grasp the white mug:** Approach the white mug. Position the gripper around the body or handle of the mug and close the gripper to grasp it securely.
2.  **Lift the white mug:** Move the gripper vertically upwards, lifting the mug clear of the table surface and the other mugs.
3.  **Move the white mug to the right plate:** Move the gripper horizontally, carrying the mug across the table until it is positioned directly above the plate on the *right*.
4.  **Place the white mug on the right plate:** Move the gripper vertically downwards to gently place the white mug onto the surface of the right plate. Open the gripper to release the mug.

Carefully look at the video and describe the sub-task motions you see being performed by the robot and the corresponding timecodes."""
    ),
]
example_model_message = [
    types.Part.from_text(
        text="""Okay, here's a breakdown of the sub-task motions shown in the video:

1.  **00:00- 00:04 Grasp the top drawer handle:** The robot arm moves downwards and towards the open top drawer, positioning its open gripper near the handle, against the front face of the drawer. It does *not* grasp the handle itself.
2.  **00:04 - 00:07 Push the top drawer closed** The robot arm moves horizontally inwards, pushing the front face of the drawer until it is fully closed."""
    ),
    types.Part.from_text(
        text="""Okay, here's the breakdown of the sub-task motions for opening the microwave:

1.  **00:00 - 00:04 Approach the microwave door handle:** The robot arm moves downwards and towards the microwave handle on the left side of the door.
2.  **00:04 - 00:07 Pull microwave door open:** The robot arm moves horizontally outwards (to the left), pushing against the handle/door edge to swing the microwave door open."""
    ),
    types.Part.from_text(
        text="""Okay, here's the breakdown of the sub-task motions for placing the white mug on the right plate:

1.  **00:00 - 00:03 Grasp the white mug:** The robot moves its gripper downwards and positions it around the white mug then closes the gripper to grasp it.
2.  **00:03 - 00:04 Lift the white mug** The robot lifts the grasped mug vertically upwards, clear of the table.
3.  **00:04 - 00:06 Move the white mug to the right plate:** The robot moves the mug horizontally towards the right, positioning it above the right plate.
4.  **00:06 - 00:08 Place the white mug on the right plate:** The robot lowers the mug down onto the surface of the right plate and then opens its gripper to release the mug"""
    ),
]


def create_prompt(task_instruction, video, subtask_labels, zero_shot=False):
    query_prompt = (
        f"""The video is showing a robot arm performing the task "{task_instruction}" The task can be broken down into these possible sub-task motions:

{subtask_labels}

"""
        + """You need to carefully look at the video and describe the sub-task motions you see being performed by the robot and the corresponding timecodes. Follow these guidelines:
1. Start off by pointing to no more than 8 items in the image . The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000. One element a line.
2. Determine the object movement and the resulting object relations. Explain the object movements.
3. Combine motions for approaching and grasping objects. Also combine motions for placing and releasing objects. 
4. Finally, output the sub-tasks that you observe and the corresponding timecodes in the json format:  [{"sub_task": <sub-task>, "time_range": <timecodes>}, ...].
"""
    )

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
        config=types.GenerateContentConfig(temperature=0.0),
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

    with open(args.subtask_labels, "r") as f:
        subtask_labels_dict = json.load(f)

    video_tasks = []
    for mp4_file in Path(args.input_dir).glob("*.mp4"):
        traj_id = mp4_file.stem
        if metadata:
            file_metadata = metadata.get(mp4_file.name)
            task_instruction = file_metadata["task"]
            demo_id = file_metadata["demo_id"]
        else:
            task_instruction, demo_id = get_task_instruction_bridge(mp4_file)

        if traj_id in labels_json_dict:
            print(f"Skipping already processed file: {mp4_file.name}")
            continue

        if traj_id not in subtask_labels_dict:
            print(f"Warning: No subtask labels found for {traj_id}, skipping")
            continue

        subtask_labels = subtask_labels_dict.get(traj_id).get("subtask_labels")

        video_tasks.append(
            {
                "mp4_file": mp4_file,
                "traj_id": traj_id,
                "task_instruction": task_instruction,
                "demo_id": demo_id,
                "subtask_labels": subtask_labels,
            }
        )

    print(f"Labeling {len(video_tasks)} videos")
    for task in video_tasks:
        mp4_file = task["mp4_file"]
        traj_id = task["traj_id"]
        task_instruction = task["task_instruction"]
        demo_id = task["demo_id"]
        subtask_labels = task["subtask_labels"]

        print(f"Processing file: {mp4_file.name}")
        video_part = part_from_file(client, mp4_file)
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

        json_blocks = re.findall(r"```json\s*(.*?)\s*```", response.text, re.DOTALL)
        try:
            motions_list = json.loads(json_blocks[1])
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error parsing JSON from response: {e}")
            motions_list = []

        for motion in motions_list:
            if isinstance(motion["time_range"], list):
                start, end = motion["time_range"]
                if isinstance(start, str) and ":" in start:
                    motion["time_range"] = start + " - " + end
                else:
                    motion["time_range"] = f"{int_to_mmss(int(start))} - {int_to_mmss(int(end))}"

        labels_json_dict[traj_id] = {
            "demo_id": demo_id,
            "motion_labels": motions_list,
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
    parser.add_argument("--batched_inference", action="store_true", help="Use batched inference for multiple videos")
    args = parser.parse_args()

    main(args)
