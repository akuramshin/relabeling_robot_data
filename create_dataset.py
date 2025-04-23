import json
import h5py
import os
import glob
import re
import argparse
import cv2
import numpy as np


def resize_image(img, resize_size):
    assert isinstance(resize_size, tuple)
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, resize_size, interpolation=cv2.INTER_LANCZOS4)
    img_resized = np.clip(np.round(img_resized), 0, 255).astype(np.uint8)
    return img_resized


def parse_file(input_file, demo_entries, output_file, sub_task, fps=10):
    grp = output_file.create_group("data")
    grp.attrs.update(input_file["data"].attrs)

    total_len = 0
    for i, (demo, annotations) in enumerate(demo_entries):
        # motion_labels = annotations["motion_labels"]
        motion_labels = {s['sub_task'].lower() : s['time_range'] for s in annotations["motion_labels"]}
        demo_id = annotations["demo_id"]

        # if subtask_id >= len(motion_labels):
        #     print(f"  Sub-task index {subtask_id} out of range for demo {demo}.")
        #     continue
        if sub_task not in motion_labels:
            # print(f"  Sub-task \"{sub_task}"not found in demo {demo}.")
            continue
        # sub_task = sub_task.replace("*", "")

        time_range = motion_labels[sub_task]
        timecode_pattern = r"(\d\d:\d\d)"
        timecode_matches = re.findall(timecode_pattern, time_range)
        if not timecode_matches or (timecode_matches and len(timecode_matches) < 2):
            print(f"Invalid timecodes found in demo: {demo} sub-task: {sub_task}")
            continue
        start_time, end_time = timecode_matches[:2]
        start_minutes, start_seconds = map(int, start_time.split(":"))
        end_minutes, end_seconds = map(int, end_time.split(":"))

        # instruction_pattern = r"(\s[A-Za-z,;\'\"\/():\s]+(?:[a-z]+).)"
        # instruction_matches = re.findall(instruction_pattern, sub_task)
        # if not instruction_matches:
        #     raise ValueError(f"Invalid instruction found in sub-task: {sub_task}")
        # # instruction = instruction_matches[0].strip() # In the case we want to use the entire line as the instruction
        # instruction = instruction_matches[0].strip().split(":")[0]
        # instruction = instruction.lower().capitalize() + "."
        task = sub_task.replace("right", "#TEMP#").replace("left", "right").replace("#TEMP#", "left")
        instruction = task.lower().capitalize() + "."

        start_frame = (start_minutes * 60 + start_seconds) * fps
        end_frame = (end_minutes * 60 + end_seconds) * fps

        actions = input_file["data"][f"demo_{demo_id}"]["actions"][start_frame:end_frame]
        rewards = input_file["data"][f"demo_{demo_id}"]["rewards"][start_frame:end_frame]
        states = input_file["data"][f"demo_{demo_id}"]["states"][start_frame:end_frame]
        robot_states = input_file["data"][f"demo_{demo_id}"]["robot_states"][start_frame:end_frame]
        ee_states = input_file["data"][f"demo_{demo_id}"]["obs"]["ee_states"][start_frame:end_frame]
        gripper_states = input_file["data"][f"demo_{demo_id}"]["obs"]["gripper_states"][start_frame:end_frame]
        joint_states = input_file["data"][f"demo_{demo_id}"]["obs"]["joint_states"][start_frame:end_frame]
        agentview_rgb = input_file["data"][f"demo_{demo_id}"]["obs"]["agentview_rgb"][start_frame:end_frame]
        eye_in_hand_rgb = input_file["data"][f"demo_{demo_id}"]["obs"]["eye_in_hand_rgb"][start_frame:end_frame]
        dones = input_file["data"][f"demo_{demo_id}"]["dones"][start_frame:end_frame]

        dones[-1] = 1
        rewards[-1] = 1

        # Create a new group for each demo
        ep_data_grp = grp.create_group(f"demo_{i}")
        ep_data_grp.create_dataset("actions", data=actions)
        ep_data_grp.create_dataset("instruction", data=instruction)
        ep_data_grp.create_dataset("rewards", data=rewards)
        ep_data_grp.create_dataset("states", data=states)
        ep_data_grp.create_dataset("dones", data=dones),
        ep_data_grp.create_dataset("robot_states", data=robot_states)
        obs_grp = ep_data_grp.create_group("obs")
        obs_grp.create_dataset("gripper_states", data=gripper_states)
        obs_grp.create_dataset("joint_states", data=joint_states)
        obs_grp.create_dataset("ee_states", data=ee_states)
        obs_grp.create_dataset("ee_pos", data=ee_states[:, :3])
        obs_grp.create_dataset("ee_ori", data=ee_states[:, 3:])
        agentview_rgb = np.stack([resize_image(img, (256, 256)) for img in agentview_rgb])
        eye_in_hand_rgb = np.stack([resize_image(img, (256, 256)) for img in eye_in_hand_rgb])
        obs_grp.create_dataset("agentview_rgb", data=agentview_rgb)
        obs_grp.create_dataset("eye_in_hand_rgb", data=eye_in_hand_rgb)

        ep_data_grp.attrs["num_samples"] = len(agentview_rgb)
        ep_data_grp.attrs["init_state"] = states[0]
        total_len += ep_data_grp.attrs["num_samples"]

        print(f"  Sub-task {sub_task} for {demo} processed with {ep_data_grp.attrs['num_samples']} frames.")

    grp.attrs["num_demos"] = len(demo_entries)
    grp.attrs["total"] = total_len


def process_dataset(args):
    """
    Processes the dataset by iterating through the HDF5 files,
    finding corresponding entries in the JSON file,
    extracting motion labels and data.
    """

    motion_segmantations, semantic_segmentations, hdf5_dir, output_dir = (
        args.motion_segmentations,
        args.semantic_segmentations,
        args.hdf5_dir,
        args.output_dir,
    )

    with open(semantic_segmentations, "r") as f:
        subtask_data = json.load(f)

    with open(motion_segmantations, "r") as f:
        motion_segments_data = json.load(f)

    hdf5_files = glob.glob(os.path.join(hdf5_dir, "*.hdf5"))

    for hdf5_file in hdf5_files:
        base_filename = os.path.basename(hdf5_file).replace(".hdf5", "")
        print(f"Processing HDF5 file: {base_filename}...")

        # Find corresponding entries in the JSON data
        demo_entries = [
            (demo, annotations) for demo, annotations in motion_segments_data.items() if demo.startswith(base_filename)
        ]

        if not demo_entries:
            print(f"  No corresponding motion segmentation entries found for {base_filename}")
            continue

        # get the set of unique subtask labels
        subtask_labels = []
        for demo in demo_entries:
            subtask_labels.extend([s['sub_task'].lower() for s in demo[1]["motion_labels"]])
        subtask_labels = set(subtask_labels)
        print(f"  Found {len(subtask_labels)} unique subtask labels for {base_filename}.")

        # In the case we want to use the semantic segmentation labels as instructions
        # subtask_list = subtask_data[base_filename]["subtask_labels"].replace("*","").split('\n')
        # subtask_labels = list(map(lambda x: "_".join(x.split(" ")[1:]).strip("_").lower()[:-1], subtask_list))

        # for i, label in enumerate(subtask_labels):
        #     new_hdf5_filename = os.path.join(hdf5_dir, f"{label}.hdf5")

        # subtask_list = subtask_data[base_filename]["subtask_labels"].replace("*", "").split("\n")
        with h5py.File(hdf5_file, "r") as original_file:
            for i, subtask in enumerate(subtask_labels):
                hdf5_path = os.path.join(output_dir, f"{base_filename}_subtask_{i}.hdf5")

                with h5py.File(hdf5_path, "w") as subtask_out_file:
                    parse_file(original_file, demo_entries, subtask_out_file, subtask, args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write subtask trajectories to LIBERO dataset files.")
    parser.add_argument(
        "--motion-segmentations",
        type=str,
        required=True,
        help="Path to the motion segmentations JSON file.",
    )
    parser.add_argument(
        "--semantic-segmentations",
        type=str,
        required=True,
        help="Path to the semantic segmentations JSON file.",
    )
    parser.add_argument(
        "--hdf5-dir",
        type=str,
        required=True,
        help="Path to the directory containing HDF5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory for processed HDF5 files.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second of the processed videos.")
    args = parser.parse_args()
    process_dataset(args)
