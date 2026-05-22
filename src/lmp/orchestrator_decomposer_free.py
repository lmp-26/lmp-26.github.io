#!/usr/bin/env python3
"""
Orchestrator server (Decomposer-Free) that coordinates task execution.
"""

USE_QUAT = False
import os
import yaml
import time
import httpx
import base64
import uvicorn
import requests
import numpy as np
from copy import deepcopy
from dotenv import load_dotenv

from google import genai
from google.genai import types
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "orchestrator_config.yaml"), "r") as f:
    ORCH_CONFIG = yaml.safe_load(f)
if ORCH_CONFIG["use_quat"]:
    from utils import from_quat_to_yaw

ORCHESTRATOR_PORT = 7999
CAMERA_URL = "http://localhost:8010"
DMP_URL = "http://localhost:8030"
LLM_URL = "http://localhost:8031"
GUI_URL = "http://localhost:8040"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

app = FastAPI(title="Task Orchestrator (Decomposer-Free)")

class TaskRequest(BaseModel):
    task: str

class TaskResponse(BaseModel):
    success: bool

def get_latest_image(output_path: str = "camera_image.png"):
    print_gui_message(f"Requesting latest image...")
    response = requests.get(f"{CAMERA_URL}/latest/color")
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    print_gui_message(f"Saved latest image to {output_path}")

def get_video(duration_seconds: float, output_path: str = "video.mp4"):
    print_gui_message(f"Requesting {duration_seconds} second video...")
    response = requests.get(f"{CAMERA_URL}/video/{duration_seconds}")
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    try:
        import subprocess
        temp_path = output_path + ".raw.mp4"
        os.rename(output_path, temp_path)
        subprocess.run(['ffmpeg', '-i', temp_path, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast', '-y', output_path], capture_output=True)
        os.remove(temp_path)
        print_gui_message(f"Transcoded video to H.264")
    except Exception as e:
        print_gui_message(f"Transcoding failed: {e}")
    print_gui_message(f"Saved video to {output_path}")

def save_cached_video(input_path: str, output_path: str, task: str, success: bool):
    dest_folder = f"{output_path}/{'success' if success else 'failure'}"
    os.makedirs(dest_folder, exist_ok=True)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    safe_task = str(task).replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    output_filename = os.path.join(dest_folder, f"{safe_task}_{timestamp}.mp4")
    os.rename(input_path, output_filename)
    print_gui_message(f"Saved task video to {output_filename}")

def compress_image(input_path: str, output_path: str, quality: tuple = (480, 270)):
    from PIL import Image
    with Image.open(input_path) as img:
        img = img.resize(quality)
        img.save(output_path, optimize=True)

def get_scene_objects():
    response = requests.get(f"{CAMERA_URL}/find_objects")
    if response.status_code == 200:
        result = response.json()
        print_gui_message(f"Scene objects: {list(result['objects'].keys())}")
        return result["objects"], result["bbox"], result["image"]
    return None
    
def run_dmp(weights: list, target_position: list, dynamic: bool = False, height: float = 0.0, llm_monitoring: bool = False):
    payload = {"weights": weights, "goal": target_position, "dynamic": dynamic, "height": height, "llm_monitoring": llm_monitoring}
    response = requests.post(f"{DMP_URL}/run_dmp", json=payload)
    if response.status_code == 200:
        result = response.json()
        if result["success"]: return True
        print_gui_message(f"DMP failed: {result['reason']}")
    return False

def print_gui_message(message: str):
    print(message)
    requests.post(f"{GUI_URL}/api/add_log", json={"message": message})

def print_prompt_log(messages: list):
    message = deepcopy(messages)
    for m in message:
        if m["role"] == "user" and isinstance(m["content"], list):
            text = m["content"][0]["text"]
            if len(m["content"]) > 1: text += "\n[IMAGE passed] "
            m["content"] = text
    requests.post(f"{GUI_URL}/api/add_prompt", json={"messages": message})

async def identify_and_handle_subtask(current_position, prior_subtasks: list, overall_task: str, correction: str = None):
    print_gui_message("\n## Capturing latest image and video...")
    get_latest_image(output_path="./cache/image.png")
    get_video(duration_seconds=1.0, output_path="./cache/video.mp4")

    print_gui_message("\n## Generating scene description...")
    scene_result = get_scene_objects()
    if not scene_result: return "FAILED", None, None, current_position

    objects, bbox, image_base64 = scene_result
    requests.post(f"{GUI_URL}/api/yolo_result", json={"image_base64": image_base64, "mime_type": "image/png"})
    objects["table"] = [0.5, 0, 0]

    task = overall_task
    print_gui_message(f"Next task: \"{task}\"")
    requests.post(f"{GUI_URL}/api/set_subtask", json={"subtask": task})

    n_functions, n_dof = ORCH_CONFIG["n_functions"], ORCH_CONFIG["n_dof"]
    dynamic_tracking, llm_monitoring = ORCH_CONFIG["dynamic_tracking"], ORCH_CONFIG["llm_monitoring"]
    compress_image("./cache/image.png", "./cache/compressed_image.png")

    # Use LLM Service
    endpoint = "/get_initial_weights" if correction is None and not prior_subtasks else "/get_followup_weights"
    payload = {"n_functions": n_functions, "n_dof": n_dof, "movable_objects": (objects, bbox), 
               "image_path": "./cache/compressed_image.png", "task": task}
    if correction: payload["correction"] = correction
    
    response = requests.post(f"{LLM_URL}{endpoint}", json=payload)
    response.raise_for_status()
    data = response.json()
    
    weights = np.array(data["weights"])
    angle, height, goal_name, end_gripper_state = data["angle"], data["height"], data["goal_name"], data["end_gripper_state"]
    
    # Scale weights
    weights[:-1] *= 20.0; weights[-2:] *= 10.0
    
    print_prompt_log(data["messages"])
    print_gui_message(f"DMP goal: {goal_name} | Angle: {angle:.2f} | Height: {height:.2f}")

    object_position = objects.get(goal_name, current_position[:3] if current_position else [0.5, 0, 0])
    
    if USE_QUAT:
        angle = from_quat_to_yaw(angle)
    
    full_goal = list(object_position) + [angle, end_gripper_state]
    full_goal[2] += height
    full_goal[0] -= full_goal[2] * 0.3 # Approach offset
    
    print_gui_message(f"Executing with goal: {full_goal}")

    # Simulation
    try:
        sim_res = requests.post(f"{DMP_URL}/simulate_dmp", json={"weights": weights.tolist(), "goal": full_goal})
        if sim_res.status_code == 200:
            plot_b64 = base64.b64encode(sim_res.content).decode('utf-8')
            requests.post(f"{GUI_URL}/api/show_dmp_plots", json={"image_base64": plot_b64, "mime_type": "image/png"})
    except Exception as e: print_gui_message(f"Sim error: {e}")

    # Confirmation
    async with httpx.AsyncClient(timeout=300.0) as client:
        res = await client.post(f"{GUI_URL}/api/confirmation", json={"options": ["Continue", "Abort"]})
        if res.status_code == 200 and res.json().get("result", {}).get("choice") == "Abort":
            return "TASK_ABORTED", task, None, current_position

    current_time = time.time()
    if goal_name and dynamic_tracking:
        try: requests.get(f"{CAMERA_URL}/start_tracking", params={"query": goal_name}, timeout=5)
        except: pass

    run_dmp(weights.tolist(), full_goal, dynamic=dynamic_tracking, height=height, llm_monitoring=llm_monitoring)
    
    if goal_name and dynamic_tracking:
        try: requests.get(f"{CAMERA_URL}/stop_tracking", timeout=5)
        except: pass

    elapsed = time.time() - current_time
    get_video(duration_seconds=max(1, int(elapsed)), output_path="./cache/task_video.mp4")
    requests.post(f"{GUI_URL}/api/task_video", json={"file_path": "./cache/task_video.mp4"})

    # Judgment
    async with httpx.AsyncClient(timeout=300.0) as client:
        res = await client.post(f"{GUI_URL}/api/confirmation", json={"options": ["Success", "Failure", "Feedback", "Complete", "Abort"]})
        if res.status_code == 200:
            result = res.json().get("result", {})
            choice, feedback = result.get("choice"), result.get("feedback")
            if choice == "Complete": return "TASK_COMPLETE", task, None, full_goal
            if choice == "Failure": return "SUBTASK_FAILED", task, None, full_goal
            if choice == "Abort": return "TASK_ABORTED", task, None, full_goal
            if choice == "Success": return "SUBTASK_COMPLETE", task, None, full_goal
            if choice == "Feedback": return "SUBTASK_FAILED", task, feedback, full_goal
    
    return "TASK_ABORTED", task, None, full_goal

async def improve_prior_task(overall_task: str, subtask: str, video_file: str = "./cache/task_video.mp4"):
    with open(os.path.join(SCRIPT_DIR, "prompt", "gemini_improve_prompt.yaml"), "r") as f:
        prompt = yaml.safe_load(f)["PROMPT"].replace("$$$$$", overall_task).replace("#####", str(subtask))
    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(video_file, 'rb') as f: video_bytes = f.read()
    res = client.models.generate_content(model=GEMINI_MODEL, contents=[types.Part(inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')), types.Part(text=prompt)])
    
    desc = corr = ""
    if "<description>" in res.text: desc = res.text.split("<description>")[1].split("</description>")[0].strip()
    if "<correction>" in res.text: corr = res.text.split("<correction>")[1].split("</correction>")[0].strip()
    return desc, corr

@app.post("/task", response_model=TaskResponse)
async def handle_task(request: TaskRequest):
    print_gui_message(f"Received task: {request.task}")
    requests.post(f"{LLM_URL}/refresh")
    
    correction = subtask = current_pos = None
    prior_subtasks = []

    while True:
        result, subtask, feedback, current_pos = await identify_and_handle_subtask(current_pos, prior_subtasks, request.task, correction)
        correction = None
        
        if result == "TASK_COMPLETE":
            save_cached_video("./cache/task_video.mp4", "./video", subtask, True)
            return TaskResponse(success=True)
        if result == "TASK_ABORTED":
            return TaskResponse(success=False)
        if result == "SUBTASK_COMPLETE":
            save_cached_video("./cache/task_video.mp4", "./video", subtask, True)
            prior_subtasks.append(subtask)
            requests.post(f"{LLM_URL}/refresh")
        if result == "SUBTASK_FAILED":
            if not feedback: _, feedback = await improve_prior_task(request.task, subtask)
            correction = feedback
            save_cached_video("./cache/task_video.mp4", "./video", subtask, False)

@app.get("/health")
async def health(): return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=ORCHESTRATOR_PORT)
