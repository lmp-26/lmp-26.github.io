import os
import time
import glob
import hydra
import numpy as np
from termcolor import colored
from omegaconf import DictConfig

import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from cameras_subscriber import CamerasSubscriber

from utils import ObservationBuffer, preprocess_video, Keyboard
from pi0 import PI0

ORCHESTRATOR_PORT = 7999

ROBOT_URL = "http://localhost:8030"
GUI_URL = "http://localhost:8040"  # NiceGUI built-in server
CAMERA_URL = "http://localhost:8010"

app = FastAPI(title="Task Orchestrator")

class TaskRequest(BaseModel):
    task: str

class TaskResponse(BaseModel):
    success: bool

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
    
def get_video(duration_seconds: float, output_path: str = "video.mp4"):
    """Request and save a video of specified duration"""
    print_gui_message(f"Requesting {duration_seconds} second video...")

    response = requests.get(f"{CAMERA_URL}/video/{duration_seconds}")
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print_gui_message(f"Saved video to {output_path}")

def open_gripper():
    requests.post(f"{ROBOT_URL}/open_gripper_wo_delay")
    return

def close_gripper():
    requests.post(f"{ROBOT_URL}/close_gripper_wo_delay")
    return

@app.get("/health")
async def health_check():
    return {
        "status": "healthy"
    }


complete = False
@app.post("/task", response_model=TaskResponse)
def handle_task(request: TaskRequest):
    interface = Keyboard()
    global complete
    print_gui_message(f"Received task: {request.task}")
    # start cameras
    topics = ["gripper_img_rgb",
              "left_realsense_img_rgb"]
    cameras_sub = CamerasSubscriber(topics=topics,
                                   server_addr='localhost',
                                   port=8082)
    cameras_sub.start_thread()
    print_gui_message("Starting task execution...")
    complete = False

    # get current robot position
    response= requests.get(f"{ROBOT_URL}/get_joint_state")
    if response.status_code == 200:
        state = response.json()["state"]
        start_position = state
    assert len(start_position) == 8, f"Expected 8 values for joint positions and gripper state, got {len(start_position)}"
    # print(f"Current robot state: {start_position}")

    obs = {}
    frames = cameras_sub.get_last_obs()
    obs["image"] = preprocess_video(np.array(frames["left_realsense_img_rgb"][-1]), specs={"resize": (224,224)})
    obs["image_gripper"] = preprocess_video(np.array(frames["gripper_img_rgb"][-1]), specs={"resize": (224,224)})
    obs['agent_pos'] = np.array(start_position)

    obs_buffer = ObservationBuffer(buffer_size=4, init_obs=obs)

    # main policy
    policy = PI0(device="cuda", action_scale=14, name="pi0", prompt=request.task)
    step_time = 1 / 20
    init_time = time.time()
    start_time = time.time()
    while True:
        try:
            curr_time = time.time()
            if interface.s_pressed:
                elapsed_seconds = time.time() - init_time
                timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
                output_path = f"./video/{request.task.replace(' ', '_')}"
                os.makedirs(output_path, exist_ok=True)
                get_video(duration_seconds=max(1, int(elapsed_seconds)), output_path=f"{output_path}/{timestamp}.mp4")
                break
            if curr_time - start_time >= step_time:
                start_time = curr_time

                obs = {}
                frames = cameras_sub.get_last_obs()
                obs["image"] = preprocess_video(np.array(frames["left_realsense_img_rgb"][-1]), specs={"resize": (224,224)})
                obs["image_gripper"] = preprocess_video(np.array(frames["gripper_img_rgb"][-1]), specs={"resize": (224,224)})
                response = requests.get(f"{ROBOT_URL}/get_joint_state")
                if response.status_code == 200:
                    state = response.json()["state"]
                    obs['agent_pos'] = np.array(state)
                # print("Current robot state:", obs['agent_pos'])
                obs_buffer.add(obs)
                obs = obs_buffer.get_buffer()

                action = policy.get_action(obs)

                if np.all(action == 0): # if all action is 0, skip
                    continue
                if action is None:
                    print("Policy done, exiting")
                    break
                print(f"Action from policy: {action}")

                assert len(action) == 8
                # if action[-1] >= 0.4: 
                if action[-1] <= -0.8:
                    close_gripper()
                elif action[-1] > 0.8:
                    open_gripper()
                requests.post(f"{ROBOT_URL}/execute_action", json={"action": action[:-1].tolist()})
                
        except KeyboardInterrupt:
            break
                    
    return TaskResponse(success=True)



def main() -> None:
    print_gui_message(f"Starting orchestrator on port {ORCHESTRATOR_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=ORCHESTRATOR_PORT)
    
if __name__ == "__main__":
    main()