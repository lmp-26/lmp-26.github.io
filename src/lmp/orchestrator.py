"""
Orchestrator server that receives tasks and coordinates their execution.
"""

USE_QUAT = False
import re
import os
import yaml
import time
import httpx
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
YOLO_URL = "http://localhost:8020"
DMP_URL = "http://localhost:8030"
LLM_URL = "http://localhost:8031"
GUI_URL = "http://localhost:8040"  # NiceGUI built-in server

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(response.content)

    print_gui_message(f"Saved latest image to {output_path}")

def get_video(duration_seconds: float, output_path: str = "video.mp4"):
    """Request and save a video of specified duration"""
    print_gui_message(f"Requesting {duration_seconds} second video...")

    response = requests.get(f"{CAMERA_URL}/video/{duration_seconds}")
    response.raise_for_status()

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(response.content)

    # Transcode to H.264 for browser compatibility if ffmpeg is available
    try:
        import subprocess
        temp_path = output_path + ".raw.mp4"
        os.rename(output_path, temp_path)
        subprocess.run([
            'ffmpeg', '-i', temp_path, 
            '-vcodec', 'libx264', 
            '-pix_fmt', 'yuv420p', 
            '-preset', 'ultrafast',
            '-y', output_path
        ], capture_output=True)
        os.remove(temp_path)
        print_gui_message(f"Transcoded video to H.264 for GUI compatibility")
    except Exception as e:
        print_gui_message(f"Transcoding failed (falling back to raw): {e}")

    print_gui_message(f"Saved video to {output_path}")

def save_cached_video(input_path: str, output_path: str, task: str, success: bool):
    """Save the video of the task attempt from cache."""
    if success:
        dest_folder = f"{output_path}/success"
    else:
        dest_folder = f"{output_path}/failure"

    os.makedirs(dest_folder, exist_ok=True)
    # get time in year_month_day_hour_minute_second format
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    output_path = os.path.join(dest_folder, f"{task}_{timestamp}.mp4")
    os.rename(input_path, output_path)
    print_gui_message(f"Saved task video to {output_path}")


def compress_image(input_path: str, output_path: str, quality: tuple = (480, 270)):
    """Compress the image to the specified quality"""
    from PIL import Image

    with Image.open(input_path) as img:
        img = img.resize(quality)
        img.save(output_path, optimize=True)

def get_scene_objects():
    """Request and print_gui_message a description of the scene"""
    response = requests.get(f"{CAMERA_URL}/find_objects")
    if response.status_code == 200:
        result = response.json()
        objects = result["objects"]
        bbox = result["bbox"]
        
        # for oid, oname in enumerate(result["labels"][0]):
        #     objects[oname.lower()] = result["positions"][0][int(oid)]

        print_gui_message(f"Scene objects: {list(objects.keys())}")
        return objects, bbox, result["image"]
    else:
        print_gui_message(f"Failed to get scene objects: {response.status_code}")
        print_gui_message(f"Response: {response.text}")
        return None

def er_get_next_task(scene_objects: list, prior_tasks: list, overall_task: str):
    """Request and print_gui_message the next task from Gemini ER"""

    with open(os.path.join(SCRIPT_DIR, "prompt", "subtask_prompt.yaml"), "r") as f:
        task_prompt = yaml.safe_load(f)["PROMPT"]

    payload = """
            You are a helpful assistant that proposes high-level subtasks for robot manipulation based on video evidence.
            Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>.
            $$$$$
              """
    payload = payload.replace("$$$$$", task_prompt)

    # Replace OBJ_LIST and PRIOR_TASKS in the payload with the actual values
    obj_list_str = "\n- ".join(scene_objects)
    prior_tasks_str = "\n- ".join(prior_tasks)

    if prior_tasks_str == "":
        prior_tasks_str = "None"
    else:
        prior_tasks_str = f"- {prior_tasks_str}"

    if obj_list_str == "":
        obj_list_str = "None"
    else:
        obj_list_str = f"- {obj_list_str}"

    payload = payload.replace("OBJ_LIST", obj_list_str)
    payload = payload.replace("PRIOR_TASKS", prior_tasks_str)
    payload = payload.replace("OVERALL_TASK", overall_task)

    print_gui_message("Querying Gemini ER for next task...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    print_gui_message(f"Using video file: ./cache/video.mp4")
    with open("./cache/video.mp4", 'rb') as f:
        video_bytes = f.read()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=types.Content(
            parts=[
                types.Part(text=payload),
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                )
            ]
        ),
        config = types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )

    think = ""
    answer = ""
    response_text = response.text
    print(response_text)
    
    # Try to extract content between <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Fallback 1: If </think> is present, take everything after it
        if "</think>" in response_text:
            answer = response_text.split("</think>")[-1].strip()
        else:
            # Fallback 2: Just use the whole text but try to remove <think> block
            answer = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()
            
    # Remove any remaining <answer> or </answer> tags just in case
    answer = re.sub(r'</?answer>', '', answer, flags=re.IGNORECASE).strip()
    
    # print_gui_message(f"Thought process from Gemini ER: {think}")
    print_gui_message(f"Response from Gemini ER: {answer}")
    return answer
    
def make_task_pretty(task: str):
    if "REACH" in task:
        if "(" in task and ")" in task:
            return "Reach for and pick up the " + task.split("(")[1].split(")")[0].lower() + "."+ (",".join(task.split(",")[1:]) if "," in task else "")
        if " " in task:
            return "Reach for and pick up the " + task.split(" ")[1].lower() + "." + (",".join(task.split(",")[1:]) if "," in task else "")
    elif "CARRY" in task:
        if "(" in task and ")" in task:
            return "Carry the " + task.split("(")[1].split(")")[0].lower() + "." + (",".join(task.split(",")[1:]) if "," in task else "")
        else:
            return "Carry the " + task.split(" ")[1].lower() + "." + (",".join(task.split(",")[1:]) if "," in task else "")
    elif "RELEASE" in task:
        return "Release the held object."
    elif "WIPING" in task:
        if "(" in task and ")" in task:
            return "Wipe the " + task.split("(")[1].split(")")[0].lower() + "."
        else:
            return "Wipe the " + task.split(" ")[1].lower() + "."
    elif "PUSH" in task:
        if "(" in task and ")" in task:
            return "Push the " + task.split("(")[1].split(")")[0].lower() + "." + (",".join(task.split(",")[1:]) if "," in task else "")
        else:
            return "Push the " + task.split(" ")[1].lower() + "." + (",".join(task.split(",")[1:]) if "," in task else "")
    return task

def get_task_name_from_task(task: str):
    if "REACH" in task:
        return "reach"
    elif "CARRY" in task:
        return "carry"
    elif "RELEASE" in task:
        return "release"
    elif "WIPING" in task:
        return "wiping"
    elif "PUSH" in task:
        return "push"
    else:
        return task.split(' ')[0].lower()
        # return "ERROR_THE_TASK_NAME_COULD_NOT_BE_DETERMINED"
    
def get_object_name_from_task(task: str):
    if "REACH" in task:
        content = None

        # Extract content between parentheses if present
        if "(" in task and ")" in task:
            content = task.split("(")[1].split(")")[0]
        # Extract content between angle brackets if present
        elif "<" in task and ">" in task:
            content = task.split("<")[1].split(">")[0]
        # Otherwise extract the word after the command
        elif " " in task:
            content = task.split(" ")[1]

        if content:
            # Remove any remaining angle brackets from the content
            content = content.replace("<", "").replace(">", "").replace("_", " ")
            return content
        
    elif "CARRY" in task or "PUSH" in task:
        content = None

        # Extract LAST content between parentheses if present
        if "(" in task and ")" in task:
            content = task.split("(")[-1].split(")")[0]

        # Extract LAST content between angle brackets if present
        elif "<" in task and ">" in task:
            content = task.split("<")[-1].split(">")[0]

        # Otherwise extract the LAST word after the command
        elif " " in task:
            content = task.split(" ")[-1]

        if content:
            # Remove any remaining angle brackets from the content
            content = content.replace("<", "").replace(">", "").replace("_", " ")
            return content.lower()
        
    elif "WIPING" in task:
        content = None

        # Extract content between parentheses if present
        if "(" in task and ")" in task:
            content = task.split("(")[1].split(")")[0]
        # Extract content between angle brackets if present
        elif "<" in task and ">" in task:
            content = task.split("<")[1].split(">")[0]
        # Otherwise extract the word after the command
        elif " " in task:
            content = task.split(" ")[1]

        if content:
            # Remove any remaining angle brackets from the content
            content = content.replace("<", "").replace(">", "").replace("_", " ")
            return content.lower()

        
def run_dmp(weights: np.ndarray | list, dmp_type: str, target_position: np.ndarray | list, dynamic: bool = False, height: float = 0.0, llm_monitoring: bool = False):
    """
    weights: list
    """
    if isinstance(weights, np.ndarray):
        weights = weights.tolist()
    if isinstance(target_position, np.ndarray):
        target_position = target_position.tolist()
    payload = {"weights": weights, "goal": target_position, "dynamic": dynamic, "height": height, "llm_monitoring": llm_monitoring}
    response = requests.post(f"{DMP_URL}/run_dmp", json=payload)
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            return True
        else:
            print_gui_message(f"DMP executed but failed due to {result['reason']}")
            return False
    print_gui_message(f"Failed to run DMP: {response.status_code}")
    print_gui_message(f"Response: {response.text}")
    return False

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

async def identify_and_handle_subtask(init_conversation, messages, current_position, prior_subtasks: list, overall_task: str, correction: str = None, dynamic_tracking: bool = True):
    # First off, we need a picture of the scene and a 1 second video clip
    # We will use the camera to capture the picture and video
    print_gui_message("\n## Capturing latest image and video...")
    get_latest_image(output_path="./cache/image.png")
    get_video(duration_seconds=1.0, output_path="./cache/video.mp4")

    # Next, let's ask LangSAM to generate a description of the scene
    print_gui_message("\n## Generating scene description...")
    scene_result = get_scene_objects()
    if scene_result is None:
        return False, "Failed to get scene objects from camera server.", "Please ensure camera server is running.", current_position, init_conversation, messages
    
    objects, bbox, image_base64 = scene_result
    print_gui_message(f"Objects: {', '.join(objects.keys())}")
    # USE GUI API /api/yolo_result to show the detected objects on the GUI
    response = requests.post(f"{GUI_URL}/api/yolo_result", json={"image_base64": image_base64, "mime_type": "image/png"})
    if response.status_code == 200:
        pass
    else:
        print_gui_message(f"Failed to display YOLO result: {response.status_code}")

    # we ask Gemini ER to tell us what we should be doing next
    print_gui_message("\n## Asking Gemini Robotics ER what to do next...")
    objects["table"] = [0.5, 0, 0]  
    task = er_get_next_task(list(objects.keys()), prior_subtasks, overall_task)
    if task is None:
        print_gui_message("Failed to get next task from Gemini ER. Aborting task.")
        return False, None, None, current_position, init_conversation, messages
    
    task_name = get_task_name_from_task(task)
    task_file_path = f"./prompt/task_library/{task_name}.yaml"
    print_gui_message(f"Task name: {task_name}")

    task_object = None
    task_object = get_object_name_from_task(task)
    print_gui_message(f"Task object: {task_object}")

    task = make_task_pretty(task)
    print_gui_message(f"Next task: \"{task}\"")
    response = requests.post(f"{GUI_URL}/api/set_subtask", json={"subtask": task})
    if response.status_code == 200:
        pass
    else:
        print_gui_message(f"Failed to set subtask: {response.status_code}")

    print_gui_message("\n## Generating DMP for task...")
    n_functions = ORCH_CONFIG["n_functions"]
    n_dof = ORCH_CONFIG["n_dof"]
    dynamic_tracking = ORCH_CONFIG["dynamic_tracking"]
    llm_monitoring = ORCH_CONFIG["llm_monitoring"]
    object_name = task_object
    object_position = None if object_name is None else objects[object_name]
    # Compress the image
    compress_image(input_path="./cache/image.png", output_path="./cache/compressed_image.png")

    # weights = generate_dmp(n_functions, n_dof, all_objects, object_name, task, correction)
    if init_conversation:
        payload = {
            "task": task,
            "n_functions": n_functions,
            "n_dof": n_dof,
            "target_object": object_name,
            "movable_objects": (objects, bbox),
            "image_path": "./cache/compressed_image.png",
            "task_file": task_file_path
        }
        response = requests.post(f"{LLM_URL}/get_initial_weights", json=payload)
    else:
        payload = {
            "correction": correction,
            "n_functions": n_functions,
            "n_dof": n_dof,
            "movable_objects": (objects, bbox),
            "image_path": "./cache/compressed_image.png"
        }
        response = requests.post(f"{LLM_URL}/get_followup_weights", json=payload)
    response.raise_for_status()
    data = response.json()
    weights = np.array(data["weights"])
    angle = data["angle"]
    height = data["height"]
    messages = data["messages"]
    if init_conversation:
        init_conversation = False
    weights[:-1] = weights[:-1] * 20.0  # Scale the position weights appropriately
    weights[-2] = weights[-2] * 10.0  # Scale angle weight appropriately
    weights[-1] = weights[-1] * 10.0  # Scale weights appropriately
    print_prompt_log(messages)
    print_gui_message(f"DMP weights: {weights}")
    print_gui_message(f"DMP Shape: {weights.shape}")
    print_gui_message(f"Proposed angle (radians): {angle}")
    print_gui_message(f"Proposed height: {height}")
    object_position = current_position[:3] if object_position is None else object_position
    if USE_QUAT:
        angle = from_quat_to_yaw(angle)
    object_position.append(angle)
    object_position.append(int(task_name in ("reach", "carry", "wiping", "push")))  
    # offsets
    object_position[2] = object_position[2] + height
    print(f"Current position: {current_position}")
    print_gui_message(f"DMP goal: {object_position}")

    print_gui_message("\n## Executing task...")

    # Call the /simulate_dmp endpoint to simulate the DMP first
    print_gui_message("Simulating DMP trajectory...")
    try:
        simulate_payload = {"weights": weights.tolist(), "goal": object_position}
        simulate_response = requests.post(f"{DMP_URL}/simulate_dmp", json=simulate_payload)

        if simulate_response.status_code == 200:
            # Got the plot image, now send it to the GUI
            import base64
            plot_image_base64 = base64.b64encode(simulate_response.content).decode('utf-8')

            gui_response = requests.post(
                f"{GUI_URL}/api/show_dmp_plots",
                json={"image_base64": plot_image_base64, "mime_type": "image/png"}
            )

            if gui_response.status_code == 200:
                print_gui_message("DMP simulation plot displayed on GUI")
            else:
                print_gui_message(f"Failed to display DMP plot on GUI: {gui_response.status_code}")
        else:
            print_gui_message(f"Failed to simulate DMP: {simulate_response.status_code}")
            print_gui_message(f"Response: {simulate_response.text}")
    except Exception as e:
        print_gui_message(f"Error during DMP simulation: {str(e)}")

    # USE GUI API /api/confirmation with options to get user approval before continuing here
    async with httpx.AsyncClient(timeout=300.0) as client:
        print_gui_message("Requesting user confirmation to proceed...")
        print(f"[DEBUG ORCHESTRATOR] Sending POST to {GUI_URL}/api/confirmation")
        confirmation_response = await client.post(f"{GUI_URL}/api/confirmation", json={"options": ["Continue", "Abort"]})
        print(f"[DEBUG ORCHESTRATOR] Received response")
        print_gui_message(f"Received confirmation response with status code: {confirmation_response.status_code}")
        if confirmation_response.status_code == 200:
            result = confirmation_response.json()
            user_choice_data = result.get("result", {})
            user_choice = user_choice_data.get("choice")
            print_gui_message(f"User chose: {user_choice}")
            if user_choice == "Abort":
                print_gui_message("Task aborted by user.")
                return "TASK_ABORTED", task, None, current_position, init_conversation, messages
        else:
            print_gui_message(f"Failed to get user confirmation: {confirmation_response.status_code}")
            return "TASK_ABORTED", task, None, current_position, init_conversation, messages

    current_time = time.time()  
    print_gui_message("Running DMP...")
    print_gui_message(f"[DEBUG] Using transformed goal at {object_position}")

    if task_object and dynamic_tracking:
        print_gui_message(f"Initializing tracking for {task_object}...")
        try:
            requests.get(f"{CAMERA_URL}/start_tracking", params={"query": task_object}, timeout=5)
        except Exception as e:
            print_gui_message(f"Failed to start tracking: {e}")

    # Start LLM monitoring before execution
    if llm_monitoring:
        print_gui_message("Starting LLM obstacle monitoring...")
        monitor_model = ORCH_CONFIG.get("monitor_model", "qwen")
        try:
            requests.post(f"{LLM_URL}/start_monitoring", json={"frequency_hz": 0.5, "monitor_model": monitor_model}, timeout=5)
            requests.get(f"{LLM_URL}/wait_for_monitoring", params={"timeout": 10.0}, timeout=15)
        except Exception as e:
            print_gui_message(f"Failed to start LLM monitoring: {e}")

    run_dmp(weights, dmp_type=task, target_position=object_position, dynamic=dynamic_tracking, height=height, llm_monitoring=llm_monitoring)
    
    # Stop LLM monitoring after execution
    if llm_monitoring:
        try:
            requests.post(f"{LLM_URL}/stop_monitoring", timeout=5)
            print_gui_message("LLM obstacle monitoring stopped.")
        except Exception as e:
            print_gui_message(f"Failed to stop LLM monitoring: {e}")

    if task_object and dynamic_tracking:
        try:
            requests.get(f"{CAMERA_URL}/stop_tracking", timeout=5)
        except Exception as e:
            print_gui_message(f"Failed to stop tracking: {e}")

    elapsed_seconds = time.time() - current_time
    print_gui_message(f"Task completed in {elapsed_seconds:.2f} seconds")
    current_position = object_position

    print_gui_message("\n## Judging task completion...")
    print_gui_message("Getting Task Video...")
    get_video(duration_seconds=max(1, int(elapsed_seconds)), output_path="./cache/task_video.mp4")
    # USE GUI API /api/task_video to show the task video on the GUI
    video_response = requests.post(f"{GUI_URL}/api/task_video", json={"file_path": "./cache/task_video.mp4"})
    if video_response.status_code == 200:
        print_gui_message(f"Task video displayed on GUI")
    else:
        print_gui_message(f"Failed to display task video: {video_response.status_code}")

    # USE GUI API /api/confirmation to figure out if task was successful
    async with httpx.AsyncClient(timeout=300.0) as client:
        success_response = await client.post(f"{GUI_URL}/api/confirmation", json={"options": ["Success", "Failure", "Feedback", "Complete","Abort"]})
        if success_response.status_code == 200:
            result = success_response.json()
            response_data = result.get("result") 

            # Handle new format with choice and feedback
            if isinstance(response_data, dict):
                user_judgment = response_data.get("choice")
                feedback_text = response_data.get("feedback")
            else:
                # Backward compatibility with old format
                user_judgment = response_data
                feedback_text = None

            print_gui_message(f"User judged task as: {user_judgment}")

            if feedback_text:
                print_gui_message(f"User feedback: {feedback_text}")
                print(f"HUMAN FEEDBACK: {feedback_text}")

            if user_judgment == "Complete":
                print_gui_message("Task marked as complete by user.")
                pass
            if user_judgment == "Failure":
                print_gui_message("Task failed. Returning false.")
                return "SUBTASK_FAILED", task, None, current_position, init_conversation, messages
            elif user_judgment == "Abort":
                print_gui_message("Task aborted by user during judgment.")
                return "TASK_ABORTED", task, None, current_position, init_conversation, messages
            elif user_judgment == "Success":
                print_gui_message("Task succeeded.")
                return "SUBTASK_COMPLETE", task, None, current_position, init_conversation, messages
            elif user_judgment == "Feedback":
                print_gui_message("Task marked for feedback. Treating as failure for retry.")
                return "SUBTASK_FAILED", task, feedback_text, current_position, init_conversation, messages
        else:
            print_gui_message(f"Failed to get task judgment: {success_response.status_code}")
            return "TASK_ABORTED", task, None, current_position, init_conversation, messages

    print_gui_message("[DEBUG] Returning task success...")
    return "TASK_COMPLETE", task, None, current_position, init_conversation, messages

async def improve_prior_task(overall_task: str, subtask: str, video_file: str = "./cache/task_video.mp4"):
    with open(os.path.join(SCRIPT_DIR, "prompt", "gemini_improve_prompt.yaml"), "r") as f:
        gemini_prompt_template = yaml.safe_load(f)["PROMPT"]

    gemini_prompt = gemini_prompt_template.replace("$$$$$", overall_task).replace("#####", subtask)

    client = genai.Client(api_key=GEMINI_API_KEY)

    print_gui_message(f"\n## Analyzing prior task failure with Gemini Robotics ER...")
    print_gui_message(f"Using video file: {video_file}")
    with open(video_file, 'rb') as f:
        video_bytes = f.read()

    image_response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                ),
                types.Part(text=gemini_prompt)
            ]
        ),
        config = types.GenerateContentConfig(
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    # Parse out <description> and <correction>
    description = ""
    correction = ""
    response_text = image_response.text
    
    desc_match = re.search(r'<description>(.*?)</description>', response_text, re.DOTALL | re.IGNORECASE)
    if desc_match:
        description = desc_match.group(1).strip()
        
    corr_match = re.search(r'<correction>(.*?)</correction>', response_text, re.DOTALL | re.IGNORECASE)
    if corr_match:
        correction = corr_match.group(1).strip()
        
    print_gui_message(f"Failure Description: {description}")
    print_gui_message(f"Correction: {correction}")
    return description, correction

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy"
    }

@app.post("/task", response_model=TaskResponse)
async def handle_task(request: TaskRequest):
    print_gui_message(f"Received task: {request.task}")

    complete = False
    correction = None
    subtask = None
    current_position = None
    prior_subtasks = []
    
    # initialize the prompt generating llm and the initial message
    init_conversation = True
    messages = []

    while not complete:
        result, subtask, feedback, current_position, init_conversation, messages = await identify_and_handle_subtask(init_conversation, messages, current_position, prior_subtasks, request.task, correction)
        correction = None
        if result == "TASK_COMPLETE":
            save_cached_video(input_path="./cache/task_video.mp4", output_path="./video", task=subtask, success=True)
            return TaskResponse(success=True)
        elif result == "TASK_ABORTED":
            return TaskResponse(success=False)
        elif result == "SUBTASK_COMPLETE":
            save_cached_video(input_path="./cache/task_video.mp4", output_path="./video", task=subtask, success=True)
            # Refresh the remote DMP generator
            requests.post(f"{LLM_URL}/refresh")
            # Reset local state to start fresh for the next subtask
            init_conversation = True
            messages = []
            prior_subtasks.append(subtask)
        elif result == "SUBTASK_FAILED":
            if feedback is not None:
                correction = feedback
            else:   
                description, correction = await improve_prior_task(request.task, subtask)
            save_cached_video(input_path="./cache/task_video.mp4", output_path="./video", task=subtask, success=False)

    return TaskResponse(success=True)


def main():
    print_gui_message(f"Starting orchestrator on port {ORCHESTRATOR_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=ORCHESTRATOR_PORT)


if __name__ == "__main__":
    main()
