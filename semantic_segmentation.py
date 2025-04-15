from google.genai import types
from google.genai.errors import ServerError, ClientError
import cv2

import os
import time
import json
from pathlib import Path
import argparse

from utils import upload_file, client, model_name, get_task_instruction


example_files = [
    upload_file(file_name="./examples/close_cabinet.png"),
    upload_file(file_name="./examples/open_cabinet.png"),
    upload_file(file_name="./examples/open_cabinet_and_place_bowl.png"),
]
example_user_message = [
    types.Part.from_text(
        text="""You are a robot arm with a simple gripper. You are given the task \"Pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate.\" Given the task and the image showing the layout of the scene, create a plan of the possible sub-task motions you will need to perform to complete the task."""
    ),
    types.Part.from_text(
        text="""You are a robot arm with a simple gripper. You are given the task \"Open the top drawer of the cabinet.\" Given the task and the image showing the layout of the scene, create a plan of the possible sub-task motions you will need to perform to complete the task.
"""
    ),
    types.Part.from_text(
        text="""You are a robot arm with a simple gripper. You are given the task \"Close the bottom drawer of the cabinet.\" Given the task and the image showing the layout of the scene, create a plan of the possible sub-task motions you will need to perform to complete the task."""
    ),
]
example_model_message = [
    types.Part.from_text(
        text="""Okay, here is a plan outlining the sub-task motions to pick up the black bowl from the top drawer and place it on the plate:

1.  **Grasp Bowl:** Approach the black bowl located inside the top drawer. Position the gripper around the rim of the bowl and close the gripper to securely grasp it.
2.  **Lift Bowl:** Move the gripper vertically upwards, lifting the bowl clear of the top drawer.
3.  **Move Bowl to Plate:** Move the gripper horizontally, carrying the bowl over the table surface until it is positioned directly above the plate on the left.
4.  **Place Bowl on Plate:** Move the gripper vertically downwards to gently place the bowl onto the surface of the plate. Open the gripper to release the bowl."""
    ),
    types.Part.from_text(
        text="""Okay, here is a plan outlining the sub-task motions to open the top drawer of the cabinet:

1.  **Grasp Handle:** Close the gripper firmly around the top drawer handle.
2.  **Pull Drawer Open:** Move the gripper horizontally outwards, straight away from the cabinet face, pulling the drawer open. Maintain a firm grip while pulling."""
    ),
    types.Part.from_text(
        text="""Okay, here is a plan outlining the sub-task motions to close the bottom drawer of the cabinet:

1.  **Grasp Handle:** Move the gripper to the handle of the *bottom* drawer. Position the gripper around the handle and close it firmly.
2.  **Push Drawer Closed:** Move the gripper horizontally inwards, towards the cabinet face, pushing the drawer until it is fully closed. Maintain a firm grip while pushing."""
    ),
]


def get_first_frame(video_path):
    vidcap = cv2.VideoCapture(str(video_path))
    success, image = vidcap.read()
    if success:
        name = str(video_path.with_name(f"{video_path.stem}_first_frame.png"))
        cv2.imwrite(name, image)
        return name
    else:
        raise ValueError(f"Failed to read video file {video_path}.")


def create_prompt(task_instruction, frame, zero_shot=False):
    base_prompt = f"""You are a robot arm with a simple gripper. You are given the task "{task_instruction}" Given the task and the image showing the layout of the scene, create a plan of the possible sub-task motions you will need to perform to complete the task."""

    if zero_shot:
        query_prompt = base_prompt + """\n\nHere is a plan outlining the sub-task motions to move the spoon to the right:
        
1. Grasp Spoon: Approach the spoon and use the gripper to grasp the spoon by the handle.
2. Lift Spoon: Move the gripper vertically upwards a small distance, just enough to lift the spoon clear of the table surface.
3. Move Right: Move the gripper horizontally to the right, carrying the spoon over the table surface to the desired new position.
4. Release Spoon: Move the gripper vertically downwards to place the spoon gently onto the table surface at the new location and open the gripper to release the spoon."""
        return [types.Content(role="user", parts=[frame, types.Part(text=query_prompt)])]
    
    query_prompt = base_prompt + "\n\nLook at the previous examples for the preferred format."
    
    contents = [
        types.Content(role="user", parts=[example_files[0], example_user_message[0]]),
        types.Content(role="model", parts=[example_model_message[0]]),
        types.Content(role="user", parts=[example_files[1], example_user_message[1]]),
        types.Content(role="model", parts=[example_model_message[1]]),
        types.Content(role="user", parts=[example_files[2], example_user_message[2]]),
        types.Content(role="model", parts=[example_model_message[2]]),
        types.Content(role="user", parts=[frame, types.Part.from_text(text=query_prompt)]),
    ]
    return contents


def extract_subtask_labels(frame_part, task_instruction):
    messages = []

    messages = create_prompt(task_instruction, frame_part)

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
        with open(labels_json_out_path, "r") as f:
            labels_json_dict = json.load(f)

    # metadata_path = Path(args.input_dir) / "metadata.json"
    # if not metadata_path.exists():
    #     raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    # with open(metadata_path) as f:
    #     metadata = json.load(f)

    seen_instructions = set()
    for mp4_file in Path(args.input_dir).glob("*.mp4"):
        traj_filename = mp4_file.stem.split("_demo")[0] + "_demo"

        if not mp4_file.exists():
            print(f"Warning: Video file {mp4_file} not found, skipping")
            continue

        # file_metadata = metadata.get(traj_filename)
        # task_instruction = file_metadata["task"]
        scene, task_instruction, _ = get_task_instruction(mp4_file)
        grounded_instruction = scene + "_" + task_instruction
        if grounded_instruction in seen_instructions:
            print(f"Skipping duplicate task instruction: {task_instruction} in scene {scene}")
            continue
        seen_instructions.add(grounded_instruction)

        if traj_filename in labels_json_dict:
            print(f"Skipping already processed file: {mp4_file}")
            continue

        print(f"Processing file: {mp4_file}")
        time.sleep(5)  # Avoid rate limiting
        first_frame = get_first_frame(mp4_file)
        first_frame_part = upload_file(first_frame)
        os.remove(first_frame)
        try:
            response = extract_subtask_labels(first_frame_part, task_instruction)
        except ServerError as e:
            print(f"Error processing file {mp4_file}: {e}")
            print("Retrying...")
            time.sleep(10)
            try:
                response = extract_subtask_labels(first_frame_part, task_instruction)
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
