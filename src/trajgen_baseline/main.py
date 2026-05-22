import os
import time
import uvicorn
import requests
from io import StringIO
from openai import OpenAI
from copy import deepcopy
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from contextlib import redirect_stdout

from main_prompt import MAIN_PROMPT
from print_output_prompt import PRINT_OUTPUT_PROMPT

ORCHESTRATOR_PORT = 7999

CAMERA_URL = "http://localhost:8010"
YOLO_URL = "http://localhost:8020" 
ROBOT_URL = "http://localhost:8030"
GUI_URL = "http://localhost:8040"  

load_dotenv()
GPT_API_KEY = os.getenv("GPT_API_KEY")
GPT_URL = os.getenv("GPT_URL") if os.getenv("GPT_URL") != "None" else None
GPT_MODEL = os.getenv("GPT_MODEL")

app = FastAPI(title="Task Orchestrator")

class TaskRequest(BaseModel):
    task: str


class TaskResponse(BaseModel):
    success: bool

def get_latest_image(output_path: str = "camera_image.png"):
    """Request and save the most recent image"""
    print_gui_message(f"Requesting latest image...")

    response = requests.get(f"{CAMERA_URL}/latest/color")
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print_gui_message(f"Saved latest image to {output_path}")

def get_video(duration_seconds: float, output_path: str = "video.mp4"):
    """Request and save a video of specified duration"""
    print_gui_message(f"Requesting {duration_seconds} second video...")

    response = requests.get(f"{CAMERA_URL}/video/{duration_seconds * (2 / 3)}")
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print_gui_message(f"Saved video to {output_path}")

def compress_image(input_path: str, output_path: str, quality: tuple = (480, 270)):
    """Compress the image to the specified quality"""
    from PIL import Image

    with Image.open(input_path) as img:
        img = img.resize(quality)
        img.save(output_path, optimize=True)

def get_scene_objects(object_or_object_part: str):
    """Request and print_gui_message a description of the scene"""
    params = {"query": object_or_object_part}
    response = requests.get(f"{CAMERA_URL}/find_objects", params=params)
    if response.status_code == 200:
        result = response.json()
        objects = result["objects"]
        bbox = result["bbox"]
        return objects, bbox, result["image"]
    else:
        print_gui_message(f"Failed to get scene objects: {response.status_code}")
        print_gui_message(f"Response: {response.text}")
        return None


def print_gui_message(message: str):
    """Send a log message to the GUI"""
    print(message)
    # USE GUI API /api/add_log to send the message
    payload = {"message": message}
    response = requests.post(f"{GUI_URL}/api/add_log", json=payload)
    if response.status_code == 200:
        return True
    else:
        print(f"Failed to send log to GUI: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
def print_prompt_log(messages: list):
    message = deepcopy(messages)
    for dict in message:
        if dict["role"] == "user":
            text = dict["content"][0]["text"]
            if len(dict["content"]) > 1 and dict["content"][1]["type"] == "image_url":
                text += "\n[IMAGE passed] "
            dict["content"] = text
    payload = {"messages": message}  
    response = requests.post(f"{GUI_URL}/api/add_prompt", json=payload)
    if response.status_code == 200:
        return True
    else:
        print(f"Failed to send prompt log to GUI: {response.status_code}")
        print(f"Response: {response.text}")
        return False


def infer(messages: list, client: OpenAI):
    new_output = ""
    completion = client.chat.completions.create(
        model=GPT_MODEL,
        temperature = 0,
        messages = messages,
        stream=True
    )

    for chunk in completion:
        chunk_content = chunk.choices[0].delta.content
        finish_reason = chunk.choices[0].finish_reason
        if chunk_content is not None:
            print(chunk_content, end="")
            new_output += chunk_content
        else:
            print("finish_reason:", finish_reason)
    return new_output

def detect_object(object_or_object_part: str):
    objects, bbox, image_base64 = get_scene_objects(object_or_object_part)
    response = requests.post(f"{GUI_URL}/api/yolo_result", json={"image_base64": image_base64, "mime_type": "image/png"})
    if response.status_code == 200:
        pass
    else:
        print_gui_message(f"Failed to display YOLO result: {response.status_code}")
    for obj_name, position in objects.items():
        # if object_or_object_part in obj_name:
        print(f"Detected object '{obj_name}' at position {position}, orientation angle from positive x-axis in xy-plane: {bbox[obj_name]['angle']:.3f} radians.")
    return
        
def execute_trajectory(trajectory: list):
    r = requests.post(f"{ROBOT_URL}/execute_trajectory", json={"trajectory": trajectory})
    r.raise_for_status()
    return

def open_gripper():
    requests.post(f"{ROBOT_URL}/open_gripper")
    return

def close_gripper():
    requests.post(f"{ROBOT_URL}/close_gripper")
    return

def task_completed():
    global complete 
    complete = True
    return
        
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy"
    }

complete = False
@app.post("/task", response_model=TaskResponse)
async def handle_task(request: TaskRequest):
    global complete
    print_gui_message(f"Received task: {request.task}")

    print_gui_message("Starting task execution...")
    complete = False

    # get current robot position
    response= requests.get(f"{ROBOT_URL}/get_state")
    if response.status_code == 200:
        state = response.json()["state"]
        print(state)
        ee_start_position = state[:3]

    new_prompt = MAIN_PROMPT.replace("[INSERT EE POSITION]", str(ee_start_position)).replace("[INSERT TASK]", request.task)
    messages = [
        {
            "role": "system",
            "content": new_prompt
        }
    ]

    # initialize gpt
    client = OpenAI(api_key=GPT_API_KEY,
                    base_url=GPT_URL)

    new_output = infer(messages, client)

    messages.append({"role":"assistant", "content":new_output})
    print_prompt_log(messages)
    print_gui_message(f"Finished generating chat bot response.")

    current_time = time.time()
    while not complete: 
        new_prompt = ""
        if len(messages[-1]["content"].split("```python")) > 1:

            code_block = messages[-1]["content"].split("```python")
            block_number = 0
            force_stop = False

            for block in code_block:
                if len(block.split("```")) > 1:
                    code = block.split("```")[0]
                    block_number += 1
                    try:
                        f = StringIO()
                        with redirect_stdout(f):
                            exec(code, globals())
                    except Exception as e:
                        print_gui_message(f"Error executing code block #{block_number}, stopping execution: {e}")
                        force_stop = True
                        break
                    else:
                        s = f.getvalue()
                        if s != "" and len(s) < 2000:
                            new_prompt += PRINT_OUTPUT_PROMPT.replace("[INSERT PRINT STATEMENT OUTPUT]", s)
                            new_prompt += "\n"
            if force_stop:
                complete = True
                break
        else:
            print_gui_message("No code block found in the response. Cannot execute.")
        
        if not complete:
            print_gui_message("Continue with task completion...")
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": new_prompt}]
            })
            new_output = infer(messages, client)
            messages.append({"role":"assistant", "content":new_output})
            print_prompt_log(messages)
    elapsed_seconds = time.time() - current_time if current_time is not None else 0
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    output_path = f"./video/{request.task.replace(' ', '_')}"
    os.makedirs(output_path, exist_ok=True)
    get_video(duration_seconds=max(1, int(elapsed_seconds)), output_path=f"{output_path}/{timestamp}.mp4")

    return TaskResponse(success=True)


def main():
    print_gui_message(f"Starting orchestrator on port {ORCHESTRATOR_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=ORCHESTRATOR_PORT)


if __name__ == "__main__":
    main()
