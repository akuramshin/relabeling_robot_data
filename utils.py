from google import genai
from google.genai import types

import os
import time
import re


def init_genai_client():
    """Initialize the Google GenAI client."""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=GEMINI_API_KEY)


def upload_file(client, file_name):
    file = client.files.upload(file=file_name)

    while file.state == "PROCESSING":
        print(f"Waiting for {file_name} to be uploaded.")
        time.sleep(10)
        file = client.files.get(name=file.name)

    if file.state == "FAILED":
        raise ValueError(file.state)
    print(f"File processing complete: " + file.uri)
    return file


def part_from_file(client, file_name):
    """Convert a file to a part for the GenAI client."""
    file = upload_file(client, file_name)
    return types.Part.from_uri(
        file_uri=file.uri,
        mime_type=file.mime_type,
    )


def get_task_instruction_bridge(video_path):
    """The task instruction is assumed to be the name of the video file without the suffix _{i}.mp4
    For example, we return "Move spoon." for "move_spoon_1.mp4"
    """
    instruction = video_path.stem
    instruction_split = instruction.split("_")

    instruction_text = " ".join(instruction_split[:-1])
    demo_number = int(instruction_split[-1])

    return instruction_text, demo_number


def get_task_instruction_libero(video_path):
    """The task instruction is assumed to be the name of the video file without the suffix _demo_{i}.mp4
    For example, we return "Move spoon." for "move_spoon_demo_1.mp4"
    """
    instruction = video_path.stem
    pattern = r"([A-Z]+(?:_[A-Z,0-9]+)*)_([a-z]+(?:_[a-z]+)*)_demo_(\d+)"

    match = re.search(pattern, instruction)

    if match:
        scene = match.group(1)
        task = match.group(2).rsplit("_", 1)[0].replace("_", " ")
        # Specific to LIBERO viewpoint
        task = task.replace("right", "#TEMP#").replace("left", "right").replace("#TEMP#", "left")
        task = task.capitalize() + "."
        demo_number = int(match.group(3))
    else:
        scene = None
        task = None
        demo_number = -1

    return scene, task, demo_number
