import json
import h5py
import os
import glob
import re


def prep_dataset(hdf5_file, attrs):
    grp = hdf5_file.create_group("data")
    grp.attrs.update(attrs)


def parse_file(input_file, demo_entries, output_file, subtask_idx):
    grp = output_file.create_group("data")
    grp.attrs.update(input_file["data"].attrs)

    total_len = 0
    for i, (demo, annotations) in enumerate(demo_entries):
        motion_labels = annotations["motion_labels"]
        demo_id = annotations["demo_id"]

        sub_task = motion_labels[subtask_idx]
        sub_task = sub_task.replace("*", "")

        timecode_pattern = r"(\d\d:\d\d)"
        timecode_matches = re.findall(timecode_pattern, sub_task)
        if timecode_matches and len(timecode_matches) < 2:
            raise ValueError(f"Invalid timecodes found in sub-task: {sub_task}")
        start_time, end_time = timecode_matches
        start_minutes, start_seconds = map(int, start_time.split(":"))
        end_minutes, end_seconds = map(int, end_time.split(":"))

        instruction_pattern = r"(\s[A-Za-z,;\'\"\/:\s]+(?:[a-z]+).)"
        instruction_matches = re.findall(instruction_pattern, sub_task)
        if not instruction_matches:
            raise ValueError(f"Invalid instruction found in sub-task: {sub_task}")
        instruction = instruction_matches[0].strip()

        start_frame = (start_minutes * 60 + start_seconds) * 10  # Assuming 10 FPS
        end_frame = (end_minutes * 60 + end_seconds) * 10

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
        obs_grp.create_dataset("agentview_rgb", data=agentview_rgb)
        obs_grp.create_dataset("eye_in_hand_rgb", data=eye_in_hand_rgb)

        ep_data_grp.attrs["num_samples"] = len(agentview_rgb)
        ep_data_grp.attrs["init_state"] = states[0]
        total_len += ep_data_grp.attrs["num_samples"]

        print(f"  Sub-task {subtask_idx} for {demo} processed with {ep_data_grp.attrs['num_samples']} frames.")

    grp.attrs["num_demos"] = len(demo_entries)
    grp.attrs["total"] = total_len


def process_dataset(motion_segmantations, semantic_segmentations, hdf5_dir, output_dir):
    """
    Processes the dataset by iterating through the HDF5 files,
    finding corresponding entries in the JSON file,
    extracting motion labels and data.
    """

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

        # In the case we want to use the semantic segmentation labels as instructions
        # subtask_list = subtask_data[base_filename]["subtask_labels"].replace("*","").split('\n')
        # subtask_labels = list(map(lambda x: "_".join(x.split(" ")[1:]).strip("_").lower()[:-1], subtask_list))

        # for i, label in enumerate(subtask_labels):
        #     new_hdf5_filename = os.path.join(hdf5_dir, f"{label}.hdf5")

        subtask_list = subtask_data[base_filename]["subtask_labels"].replace("*", "").split("\n")
        with h5py.File(hdf5_file, "r") as original_file:
            for i in range(len(subtask_list)):
                hdf5_path = os.path.join(output_dir, f"{base_filename}_subtask_{i}.hdf5")

                with h5py.File(hdf5_path, "w") as subtask_out_file:
                    parse_file(original_file, demo_entries, subtask_out_file, i)


# Example usage:
motion_responses = "/home/artur/Downloads/test_out/dataset_motion_labels.json"  # Replace with the actual path
semantic_responses = "/home/artur/Downloads/test_out/dataset_subtask_labels.json"
hdf5_dir = "/home/artur/libero_90"  # Replace with the actual path
output_dir = "/home/artur/Downloads/test_out"  # Replace with the actual path
process_dataset(motion_responses, semantic_responses, hdf5_dir, output_dir)
