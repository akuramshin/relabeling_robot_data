from google import genai
from google.genai import types

import os
import time
import re


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
client = genai.Client(api_key=GOOGLE_API_KEY)
model_name = "gemini-2.5-pro-exp-03-25"
# model_name = "gemini-2.0-flash"


def upload_file(file_name):
    file = client.files.upload(file=file_name)

    while file.state == "PROCESSING":
        print(f"Waiting for {file_name} to be uploaded.")
        time.sleep(10)
        file = client.files.get(name=file.name)

    if file.state == "FAILED":
        raise ValueError(file.state)
    print(f"File processing complete: " + file.uri)

    return types.Part.from_uri(
        file_uri=file.uri,
        mime_type=file.mime_type,
    )

def get_task_instruction(video_path):
    """The task instruction is assumed to be the name of the video file without the suffix _demo_{i}.mp4
    For example, we return "Move spoon." for "move_spoon_demo_1.mp4"
    """
    instruction = video_path.stem
    pattern = r"([A-Z]+(?:_[A-Z,0-9]+)*)_([a-z]+(?:_[a-z]+)*)_demo_(\d+)"

    match = re.search(pattern, instruction)

    if match:
        scene = match.group(1)
        task = match.group(2).replace("_", " ").capitalize() + "."
        demo_number = int(match.group(3))
    else:
        scene = None
        task = None
        demo_number = -1

    return scene, task, demo_number
