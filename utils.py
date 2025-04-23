from google import genai
from google.genai import types

from bespokelabs import curator

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


def get_task_instruction(video_path):
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


class SemanticBreakdownGenerator(curator.LLM):

    def prompt(self, input: dict) -> str:
        """Generate a prompt using the first frame and the task instruction."""
        prompt = f"""You are a robot arm with a simple gripper. You are given the task "{input['task_instruction']}" Given the task and the image showing the layout of the scene, create a plan of the possible sub-task motions you will need to perform to complete the task.

Follow these guidelines:
1. Start off by pointing to no more than 8 items in the image . The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000. One element a line.
2. Think step by step and list the locations and relations of all objects, noting any object that might interfere with the task.
3. Finally, output the sub-tasks that will need to be completed to "{input['task_instruction']}" in this scene. Try to be concise. Output in the json format:  [{"sub_task": <sub-task>, "description": <description>}, ...]

Example sub-tasks: "Grasp the bowl to the right of the white mug", "Move the mug towards the red plate", "Rotate the knob", "Pull the top drawer open", "Push the bottom drawer closed", "Place the can on top of the cabinet", "Lift the middle bowl"
        """
        return prompt, curator.types.Image(url=input["image_url"])

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""

        return {
            input["scenario_name"]: {
                "task_instruction": input["task_instruction"],
                "subtask_labels": response,
                "response": response,
            }
        }


class MotionBreakdownGenerator(curator.LLM):

    def prompt(self, input: dict) -> str:
        """Generate a prompt using the video, task instruction, and subtask labels."""
        prompt = (
            f"""The video is showing a robot arm performing the task "{input['task_instruction']}" The task can be broken down into these possible sub-task motions:

{input['subtask_labels']}

"""
            + """You need to carefully look at the video and describe the sub-task motions you see being performed by the robot and the corresponding timecodes. Follow these guidelines:
1. Start off by pointing to no more than 8 items in the image . The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000. One element a line.
2. Determine the object movement and the resulting object relations. Think about where the object and its relational objects are located in the scene on a global scale. Think step by step and list the locations and relations of all objects. Explain the object movements.
3. Finally, output the sub-tasks that you observe and the corresponding timecodes in the json format:  [{"sub_task": <sub-task>, "time_range": <timecodes>}, ...]
"""
        )
        return prompt, curator.types.File(url=input["video_file"], mime_type="video/mp4")

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input to the model into the desired output format.."""

        return {
            input["scenario_name"]: {
                "task_instruction": input["task_instruction"],
                "motion_labels": response,
                "response": response,
            }
        }
