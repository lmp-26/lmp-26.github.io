import os
import cv2
import json
import base64
import uvicorn
import requests
import threading
import asyncio
import time
import yaml
import numpy as np
from openai import OpenAI
from fastapi import FastAPI
from termcolor import colored
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Tuple, List, Optional, Any

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "prompt", "placeholder_config.yaml"), 'r') as f:
    PLACEHOLDER_CONFIG = yaml.safe_load(f)

GPT_API_KEY = os.getenv("GPT_API_KEY")
GPT_URL = os.getenv("GPT_URL") if os.getenv("GPT_URL") != "None" else None
GPT_MODEL = os.getenv("GPT_MODEL")

class DMPGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=GPT_API_KEY,
                             base_url=GPT_URL)
        self.past_weights = {}
        self.history_corrections = {}
        self.init_conversation = True
        self.messages = []
        # Monitoring state
        self.current_weights_data = {
            "weights": None,
            "angle": 0.0,
            "height": 0.0
        }
        self.reference_image_data = None
        self.last_params = {
            "n_functions": 10,
            "n_dof": 6,
            "movable_objects": None,
            "task": None
        }
        self.monitoring_thread = None
        self.stop_monitoring_evt = threading.Event()
        self.monitoring_ready = threading.Event()
        
    def load_prompt_template(self, prompt_file: str = "prompt/dmp_gen_prompt.yml"):
        """Load the prompt template from YAML file."""
        if not os.path.isabs(prompt_file):
            prompt_file = os.path.join(SCRIPT_DIR, prompt_file)

        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        with open(prompt_file, 'r') as f:
            return yaml.safe_load(f)


    def fill_template(self, template: str, config: dict) -> str:
        """Fill in template placeholders with configuration values."""
        result = template

        # Calculate total weights
        total_weights = config["dimensions"] * config["num_functions"]

        # Replace all placeholders
        replacements = {
            "${.dimensions}": str(config["dimensions"]),
            "${.num_functions}": str(config["num_functions"]),
            "${.total_weights}": str(total_weights),
            "${.goal_obj_name}": config["goal_obj_name"],
            # "${.robot_joint_info}": config["robot_joint_info"],
            "${.robot_base_position}": config["robot_base_position"],
            "${.robot_base_orientation}": config["robot_base_orientation"],
            "${.initial_tcp_position}": config["initial_tcp_position"],
            "${.initial_tcp_velocities}": config["initial_tcp_velocities"],
            "${.movable_objects}": config["movable_objects"],
            "${.task}": config["task"]
        }

        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result
    
    def clean_object_position(self, objects) -> str:
        """
        Convert object position dict into a pretty markdown table string.
        """
        if isinstance(objects, list) or isinstance(objects, tuple):
            object_dict, bbox_dict = objects[0], objects[1]
        else:
            # Fallback if objects is not a tuple/list
            object_dict = objects
            bbox_dict = {}

        if not object_dict:
            return ""

        lines = []
        lines.append("| name | position | height | width | length | angle |")
        lines.append("|------|----------|--------|-------|--------|-------|")

        for name, pos in object_dict.items():
            if pos is None:
                pos_str = "N/A"
            else:
                pos_str = ", ".join(f"{p:.3f}" for p in pos)
            bbox = bbox_dict.get(name, None)
            if bbox is not None:
                height = f"{bbox['height']:.3f}"
                width = f"{bbox['width']:.3f}"
                length = f"{bbox['length']:.3f}"
                angle = f"{bbox['angle']:.3f}"
            else:
                height = "N/A"
                width = "N/A"
                length = "N/A"
                angle = "N/A"
            lines.append(f"| {name} | {pos_str} | {height} | {width} | {length} | {angle} |")

        table = "\n".join(lines)

        return f"""{table}"""
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    
    def clean_content(self, content: str, image_path: str | None = None):
        if not image_path:
            return content
        else:
            base64_image = self.encode_image(image_path)
            clean_content = [
                {"type": "text", "text": content},
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
            return clean_content

    def generate_initial_message(self, 
                                task: str,
                                n_functions: int = 10,
                                n_dof: int = 6,
                                target_object: str = "target object",
                                movable_objects: Tuple[dict, dict] = ({"target object" : [0.0, 0.0, 0.0]}, {}),
                                image_path: str | None = None,
                                config: dict | None = None,
                                prompt_file: str | None= "prompt/dmp_gen_prompt.yml",
                                task_file: str | None = None):
        """
        Generate the initial message for DMP weight generation.

        Args:
            task_description: Description of the task to perform
            n_functions: Number of basis functions for the DMP
            n_dof: Number of degrees of freedom (dimensions)
            config: Optional configuration dictionary with scene/robot info
            prompt_file: Path to the YAML prompt template file

        Returns:
            List of message dictionaries in OpenAI chat format
        """
        # Use provided config or fall back to placeholders
        if config is None:
            config = PLACEHOLDER_CONFIG.copy()

        # Update with provided parameters
        config["dimensions"] = n_dof
        config["num_functions"] = n_functions
        config["goal_obj_name"] = target_object if target_object else config["goal_obj_name"]
        config["task"] = task
        config["movable_objects"] = self.clean_object_position(movable_objects)

        # Load prompt templates
        prompt_template = self.load_prompt_template(prompt_file)
        task_template = self.load_prompt_template(task_file)

        # Fill in the system prompt
        system_prompt = self.fill_template(prompt_template["PROMPT_SYSTEM"], config)

        # Fill in the task description
        task_prompt = self.fill_template(prompt_template["task_description"], config)

        # Fill in the initial prompt
        initial_prompt = self.fill_template(prompt_template["PROMPT_INITIAL"], config)
        # initial_prompt = prompt_template["PROMPT_INITIAL"].replace("${.task_description}", task_description)

        task_spec = task_template["TASK_SPEC"]

        content = self.clean_content(task_prompt + "\n\n" + initial_prompt + "\n\n" + task_spec, image_path)

        # Construct the message list
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ]

        return messages


    def generate_followup_message(self, 
                                action_plan: str,
                                n_functions: int = 10,
                                n_dof: int = 6,
                                movable_objects: Tuple[dict, dict] = ({"target object" : [0.0, 0.0, 0.0]}, {}),
                                image_path: str = None,
                                config: dict = None,
                                prompt_file: str = "prompt/dmp_gen_prompt.yml"):
        """
        Generate a follow-up message for iterative DMP weight refinement.

        Args:
            past_weights: Previous weight matrix as list of lists
            history: Trial history description
            action_plan: High-level action plan analysis
            n_functions: Number of basis functions for the DMP
            n_dof: Number of degrees of freedom (dimensions)
            config: Optional configuration dictionary with scene/robot info
            prompt_file: Path to the YAML prompt template file

        Returns:
            String containing the follow-up prompt
        """
        # Use provided config or fall back to placeholders
        if config is None:
            config = PLACEHOLDER_CONFIG.copy()

        # Update with provided parameters
        config["dimensions"] = n_dof
        config["num_functions"] = n_functions
        config["movable_objects"] = self.clean_object_position(movable_objects)
        self.history_corrections[f"trial {len(self.past_weights) - 1}"] = action_plan

        # Load prompt templates
        prompt_template = self.load_prompt_template(prompt_file)

        # Fill in the follow-up prompt
        followup_prompt = self.fill_template(prompt_template["PROMPT_FOLLOW_UP"], config)

        # Replace dynamic content
        followup_prompt = followup_prompt.replace("${.past_weights}", json.dumps(self.past_weights))
        followup_prompt = followup_prompt.replace("${.history}", json.dumps(self.history_corrections))
        action_plan = "This is the first trial on this subtask hence there is no past history to report." if action_plan is None else action_plan
        followup_prompt = followup_prompt.replace("${.action_plan}", action_plan)

        followup_prompt = self.clean_content(followup_prompt, image_path)

        messages = {
            "role": "user",
            "content": followup_prompt
        }

        return messages

    def get_initial_weights(self,
                            task: str,
                            n_functions: int,
                            n_dof: int,
                            target_object: str,
                            movable_objects: Tuple[dict, dict],
                            image_path: str = None,
                            task_file: str = None
                            ):
        # Save params for monitoring
        self.last_params.update({
            "task": task,
            "n_functions": n_functions,
            "n_dof": n_dof,
            "target_object": target_object,
            "movable_objects": movable_objects,
            "task_file": task_file
        })

        initial_messages = self.generate_initial_message(
            task=task,
            n_functions=n_functions,
            n_dof=n_dof,
            target_object=target_object,
            movable_objects=movable_objects,
            image_path=image_path,
            task_file=task_file
        )
        self.messages = initial_messages.copy()

        initial_response = self.client.chat.completions.create(
            model=GPT_MODEL,
            temperature=0,
            messages=initial_messages,
            # reasoning_effort= "low"
        )
        initial_response_text = initial_response.choices[0].message.content.strip()
        print(colored("[DMP Generator] Initial Response: ", "green") + initial_response_text)
        
        initial_weights, initial_angle, initial_height = self.convert_response_to_weights(
            initial_response_text, n_dof=n_dof, n_functions=n_functions
        )

        assert initial_weights is not None, "Failed to generate valid initial weights from LLM response."

        self.past_weights["trial 0"] = initial_weights.tolist()        
        self.messages.append({
            "role": "assistant",
            "content": initial_response_text
        })

        # Update current weights data
        self.current_weights_data.update({
            "weights": initial_weights.tolist(),
            "angle": initial_angle,
            "height": initial_height
        })
        return initial_weights, initial_angle, initial_height

    
    def get_followup_weights(self,
                            correction,
                            n_functions,
                            n_dof,
                            movable_objects,
                            image_path = None
                            ):
        # Update params for monitoring
        self.last_params.update({
            "n_functions": n_functions,
            "n_dof": n_dof,
            "movable_objects": movable_objects
        })

        followup_message = self.generate_followup_message(
            action_plan=correction,
            n_functions=n_functions,
            n_dof=n_dof,
            movable_objects=movable_objects,
            image_path=image_path
        )
        self.messages.append(followup_message)

        response = self.client.chat.completions.create(
            model=GPT_MODEL,
            temperature=0,
            messages=self.messages,
            # reasoning_effort= "low"
        )
        response_text = response.choices[0].message.content.strip()
        print(colored("[DMP Generator] Follow-up Response: ", "green") + response_text)
        
        weights, angle, height = self.convert_response_to_weights(
            response_text, n_dof=n_dof, n_functions=n_functions
        )

        assert weights is not None, f"Failed to generate valid follow-up weights for trial {len(self.past_weights)}."

        self.past_weights[f"trial {len(self.past_weights)}"] = weights.tolist()
        self.messages.append({
            "role": "assistant",
            "content": response_text
        })

        # Update current weights data
        self.current_weights_data.update({
            "weights": weights.tolist(),
            "angle": angle,
            "height": height
        })
        return weights, angle, height

    def start_monitoring(self, frequency_hz: float = 0.5, monitor_model: str = "qwen"):
        """Start a thread to monitor for obstacle changes."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            print(colored("[DMP Monitor] ", "yellow") + "Monitoring already running.")
            return

        self.stop_monitoring_evt.clear()
        self.monitoring_ready.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(frequency_hz, monitor_model), 
            daemon=True
        )
        self.monitoring_thread.start()
        print(colored("[DMP Monitor] ", "green") + f"Monitoring started at {frequency_hz}Hz using {monitor_model}")

    def _monitoring_loop(self, frequency_hz, monitor_model):
        interval = 1.0 / frequency_hz
        CAMERA_URL = "http://localhost:8010"
        
        # 1. Take initial image to start monitoring
        last_img_data = None
        while not self.stop_monitoring_evt.is_set() and last_img_data is None:
            try:
                response = requests.get(f"{CAMERA_URL}/latest/color", timeout=2)
                if response.status_code == 200:
                    last_img_data = response.content
                    self.monitoring_ready.set()
                    print(colored("[DMP Monitor] ", "green") + "Initial image captured, monitoring loop started.")
                else:
                    print(colored("[DMP Monitor] ", "yellow") + "Waiting for camera to be ready for monitoring...")
                    time.sleep(1)
            except Exception as e:
                print(colored("[DMP Monitor] ", "red") + f"Initial camera error: {e}")
                time.sleep(1)

        iter_num = 0
        while not self.stop_monitoring_evt.is_set():
            start_time = time.time()
            try:
                # 2. Fetch new image from Orbbec
                try:
                    response = requests.get(f"{CAMERA_URL}/latest/color", timeout=2)
                    if response.status_code != 200:
                        raise Exception("Failed to fetch image from Orbbec")
                    new_img_data = response.content
                except Exception as e:
                    print(colored("[DMP Monitor] ", "red") + f"Camera error: {e}")
                    time.sleep(interval)
                    continue

                # --- Save side-by-side comparison ---
                try:
                    prev_img = cv2.imdecode(np.frombuffer(last_img_data, np.uint8), cv2.IMREAD_COLOR)
                    curr_img = cv2.imdecode(np.frombuffer(new_img_data, np.uint8), cv2.IMREAD_COLOR)
                    if prev_img is not None and curr_img is not None:
                        h, w = prev_img.shape[:2]
                        curr_img_resized = cv2.resize(curr_img, (w, h))
                        side_by_side = cv2.hconcat([prev_img, curr_img_resized])
                        # cv2.imwrite(os.path.join(SCRIPT_DIR, "cache", f"monitor_{iter_num}.png"), side_by_side)
                    iter_num += 1
                except Exception as e:
                    print(colored("[DMP Monitor] Side-by-side save error: ", "red") + str(e))
                # ------------------------------------

                # 3. Call LLM to check for changes
                current_weights = self.current_weights_data.get("weights")
                n_functions = self.last_params.get("n_functions", 10)
                n_dof = self.last_params.get("n_dof", 6)
                
                needs_update, reason, monitor_weights = self._check_obstacle_change(
                    last_img_data, new_img_data, current_weights, 
                    n_dof=n_dof, n_functions=n_functions,
                    old_is_path=False,
                    monitor_model=monitor_model
                )

                # --- Visualization ---
                try:
                    nparr = np.frombuffer(new_img_data, np.uint8)
                    vis_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if vis_img is not None:
                        # Resize to a reasonable display size
                        vis_img = cv2.resize(vis_img, (640, 480))
                        
                        color = (0, 0, 255) if needs_update else (0, 255, 0)
                        status_text = "UPDATE REQUIRED" if needs_update else "SAFE"
                        cv2.putText(vis_img, f"Status: {status_text}", (20, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        
                        # Wrap and draw reason text
                        y_pos = 80
                        wrapped_reason = [reason[i:i+60] for i in range(0, len(reason), 60)]
                        for line in wrapped_reason:
                            cv2.putText(vis_img, line, (20, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                            y_pos += 25
                        
                        cv2.imwrite(os.path.join(SCRIPT_DIR, "cache", "monitor_view.png"), vis_img)
                        # cv2.imshow("DMP Monitor", vis_img)
                        # cv2.waitKey(1)
                except Exception as ve:
                    print(colored("[DMP Monitor] Visualization error: ", "red") + str(ve))
                # ---------------------

                if needs_update:
                    print(colored("[DMP Monitor] ", "cyan") + f"Update required: {reason}")
                    
                    # Use weights from monitor if available and valid
                    if monitor_weights is not None:
                        print(colored("[DMP Monitor] ", "green") + "Using weights provided by monitor LLM.")
                        self.current_weights_data.update({
                            "weights": monitor_weights.tolist(),
                            # Keep angle and height from previous state
                            "angle": self.current_weights_data["angle"],
                            "height": self.current_weights_data["height"]
                        })
                        self.past_weights[f"trial {len(self.past_weights)}"] = monitor_weights.tolist()
                        
                        # Only update the reference image if new weights were successfully generated
                        last_img_data = new_img_data
                        print(colored("[DMP Monitor] ", "green") + "DMP weights updated automatically and reference image updated.")
                    else:
                        print(colored("[DMP Monitor] ", "red") + "Monitor didn't provide valid weights; keeping previous reference image.")

            except Exception as e:
                print(colored("[DMP Monitor] ", "red") + f"Loop error: {e}")

            elapsed = time.time() - start_time
            print(colored("[DMP Monitor] ", "magenta") + f"Iteration took {elapsed:.2f}s (target interval: {interval:.2f}s)")
            time.sleep(max(0, interval - elapsed))
        
        # cv2.destroyWindow("DMP Monitor")

    def _check_obstacle_change(self, old_image_data, new_image_bytes, current_weights, n_dof=6, n_functions=10, old_is_path=False, monitor_model="qwen") -> Tuple[bool, str, Optional[np.ndarray]]:
        """Use Cloud API to compare images and check for obstacle changes."""
        
        def process_and_encode(data, is_path=False):
            if is_path:
                img = cv2.imread(data)
            else:
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None
            
            # Resize to 448x448 to save VRAM and improve processing speed on 5GB GPU
            img_resized = cv2.resize(img, (448, 448))
            _, buffer = cv2.imencode('.jpg', img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return base64.b64encode(buffer).decode("utf-8")

        old_b64 = process_and_encode(old_image_data, is_path=old_is_path)
        new_b64 = process_and_encode(new_image_bytes, is_path=False)

        if not old_b64 or not new_b64:
            return False, "Image processing failed", None

        # Load prompt from YAML
        prompt_path = os.path.join(SCRIPT_DIR, "prompt", "obstacle_monitor_prompt.yml")
        with open(prompt_path, "r") as f:
            monitor_prompt = yaml.safe_load(f)["PROMPT"]
        
        # Inject current weights, dimensions and task into prompt
        weights_str = json.dumps(current_weights) if current_weights is not None else "None (first trial)"
        task_str = self.last_params.get("task", "None")
        monitor_prompt = monitor_prompt.replace("${.current_weights}", weights_str)
        monitor_prompt = monitor_prompt.replace("${.n_dof}", str(n_dof))
        monitor_prompt = monitor_prompt.replace("${.n_functions}", str(n_functions))
        monitor_prompt = monitor_prompt.replace("${.task}", str(task_str))

        try:
            if "gpt" in monitor_model.lower():
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": monitor_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{old_b64}"}
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{new_b64}"}
                            }
                        ]
                    }
                ]
                
                response = self.client.chat.completions.create(
                    model=monitor_model,
                    messages=messages,
                    temperature=0,
                )
                raw_content = response.choices[0].message.content.strip()
            else:
                api_url = os.getenv("QWEN_API_BASE", "http://localhost:11434/api/chat")
                model_name = os.getenv("QWEN_MODEL", "qwen3.5:2b")
                
                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": monitor_prompt,
                            "images": [old_b64, new_b64]
                        }
                    ],
                    "stream": False,
                    "format": "json",
                    "think": False,
                    "options": {
                        "temperature": 0
                    }
                }

                response = requests.post(api_url, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    raw_content = result["message"]["content"].strip()
                else:
                    raise Exception(f"Cloud API error: {response.status_code} - {response.text}")

            print(colored(f"[DMP Monitor] {monitor_model} Response: ", "blue") + raw_content)
            
            # extract JSON from potential Markdown code blocks
            json_content = raw_content
            if "```" in raw_content:
                import re
                match = re.search(r"```(?:json)?\n?(.*?)\n?```", raw_content, re.DOTALL)
                if match:
                    json_content = match.group(1).strip()
            
            content = json.loads(json_content)
            needs_update = content.get("needs_update", False)
            reason = content.get("reason", "No reason provided")
            
            # Extract weights if present and valid
            monitor_weights, _, _ = self.convert_response_to_weights(json_content, n_dof=n_dof, n_functions=n_functions)
            
            return needs_update, reason, monitor_weights
        except Exception as e:
            print(colored("[DMP Monitor] ", "red") + f"Monitoring error: {e}")
        
        return False, "", None

    def convert_response_to_weights(self, response: str, n_dof: int = 6, n_functions: int = 10):
        """
        Convert LLM response string to DMP weights with validation.
        """
        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}")
            if start == -1 or end == -1:
                print(colored("[DMP Generator] ", "red") + "No JSON found in response.")
                return None, 0.0, 0.0

            json_response = response[start:end+1]

            # Parse JSON
            weights_dict = json.loads(json_response)

            if "weights" not in weights_dict:
                print(colored("[DMP Generator] ", "red") + "JSON missing 'weights' key.")
                return None, 0.0, 0.0

            weights = np.array(weights_dict["weights"])
            
            # Validate shape
            expected_shape = (n_dof, n_functions)
            if weights.shape != expected_shape:
                print(colored("[DMP Generator] ", "red") + f"Weight shape mismatch. Expected {expected_shape}, got {weights.shape}")
                return None, 0.0, 0.0

            angle = float(weights_dict.get("angle", 0.0))
            height = float(weights_dict.get("height", 0.0))

            return weights, angle, height
        except Exception as e:
            print(colored("[DMP Generator] ", "red") + f"Failed to convert response to weights: {e}")
            return None, 0.0, 0.0

    def convert_tasklib_to_weights(self, weight_desc):
        """Convert weight description from tasklib format to standard format."""
        raise NotImplementedError

app = FastAPI()
generator = DMPGenerator()
print(colored("[DMP Generator] ", "green") + "DMP Generator initialized and API server is ready.")

class InitialWeightsRequest(BaseModel):
    task: str
    n_functions: int
    n_dof: int
    target_object: Optional[str] = None
    movable_objects: Any
    image_path: Optional[str] = None
    task_file: Optional[str] = None

class FollowupWeightsRequest(BaseModel):
    correction: Optional[str] = None
    n_functions: int
    n_dof: int
    movable_objects: Any
    image_path: Optional[str] = None

@app.post("/get_initial_weights")
async def api_get_initial_weights(req: InitialWeightsRequest):
    weights, angle, height = generator.get_initial_weights(
        task=req.task,
        n_functions=req.n_functions,
        n_dof=req.n_dof,
        target_object=req.target_object,
        movable_objects=req.movable_objects,
        image_path=req.image_path,
        task_file=req.task_file
    )
    return {
        "weights": weights.tolist(),
        "angle": angle,
        "height": height,
        "messages": generator.messages
    }

@app.post("/get_followup_weights")
async def api_get_followup_weights(req: FollowupWeightsRequest):
    weights, angle, height = generator.get_followup_weights(
        correction=req.correction,
        n_functions=req.n_functions,
        n_dof=req.n_dof,
        movable_objects=req.movable_objects,
        image_path=req.image_path
    )
    return {
        "weights": weights.tolist(),
        "angle": angle,
        "height": height,
        "messages": generator.messages
    }

@app.post("/refresh")
async def api_refresh():
    global generator
    generator = DMPGenerator()
    return {"status": "success", "message": "DMPGenerator reinitialized"}

@app.get("/get_current_weights")
async def api_get_current_weights():
    return generator.current_weights_data

@app.get("/wait_for_monitoring")
async def api_wait_for_monitoring(timeout: float = 10.0):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if generator.monitoring_ready.is_set():
            return {"status": "ready"}
        await asyncio.sleep(0.5)
    return {"status": "timeout"}

class MonitoringRequest(BaseModel):
    frequency_hz: float = 0.5
    monitor_model: str = "qwen"

@app.post("/start_monitoring")
async def api_start_monitoring(req: MonitoringRequest):
    generator.start_monitoring(req.frequency_hz, req.monitor_model)
    return {"status": "success", "message": f"Monitoring started at {req.frequency_hz}Hz using {req.monitor_model}"}

@app.post("/stop_monitoring")
async def api_stop_monitoring():
    generator.stop_monitoring_evt.set()
    return {"status": "success", "message": "Monitoring stop signal sent"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8031)