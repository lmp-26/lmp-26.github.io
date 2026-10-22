import os
import json
import yaml
import base64
import uvicorn
import asyncio
import time
import numpy as np
from typing import Tuple, List, Optional, Any
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from termcolor import colored
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "prompt", "placeholder_config.yaml"), 'r') as f:
    PLACEHOLDER_CONFIG = yaml.safe_load(f)

GPT_API_KEY = os.getenv("GPT_API_KEY")
GPT_URL = os.getenv("GPT_URL") if os.getenv("GPT_URL") != "None" else None
GPT_MODEL = os.getenv("GPT_MODEL")


class DMPGeneratorCF:
    def __init__(self):
        self.client = OpenAI(api_key=GPT_API_KEY,
                             base_url=GPT_URL)
        self.past_weights = {}
        self.past_goal_names = {}
        self.history_corrections = {}
        self.init_conversation = True
        self.messages = []
        
    def load_prompt_template(self, prompt_file: str = "prompt/dmp_gen_prompt_composer_free.yml"):
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
        total_weights = config["dimensions"] * config["num_functions"]
        replacements = {
            "${.dimensions}": str(config["dimensions"]),
            "${.num_functions}": str(config["num_functions"]),
            "${.total_weights}": str(total_weights),
            "${.goal_obj_name}": config.get("goal_obj_name", "N/A"),
            "${.robot_base_position}": config.get("robot_base_position", "N/A"),
            "${.robot_base_orientation}": config.get("robot_base_orientation", "N/A"),
            "${.initial_tcp_position}": config.get("initial_tcp_position", "N/A"),
            "${.initial_tcp_velocities}": config.get("initial_tcp_velocities", "N/A"),
            "${.movable_objects}": config["movable_objects"],
            "${.task}": config["task"]
        }
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)
        return result
    
    def clean_object_position(self, objects) -> str:
        """Convert object position dict into a pretty markdown table string."""
        if isinstance(objects, list) or isinstance(objects, tuple):
            object_dict, bbox_dict = objects[0], objects[1]
        else:
            object_dict = objects
            bbox_dict = {}

        if not object_dict:
            return ""

        lines = ["| name | position | height | width | length | angle |",
                 "|------|----------|--------|-------|--------|-------|"]

        for name, pos in object_dict.items():
            pos_str = ", ".join(f"{p:.3f}" for p in pos) if pos is not None else "N/A"
            bbox = bbox_dict.get(name)
            if bbox:
                h, w, l, a = f"{bbox['height']:.3f}", f"{bbox['width']:.3f}", f"{bbox['length']:.3f}", f"{bbox['angle']:.3f}"
            else:
                h = w = l = a = "N/A"
            lines.append(f"| {name} | {pos_str} | {h} | {w} | {l} | {a} |")
        return "\n".join(lines)
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def clean_content(self, content: str, image_path: str = None):
        if not image_path:
            return content
        base64_image = self.encode_image(image_path)
        return [
            {"type": "text", "text": content},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]

    def generate_initial_message(self, n_functions: int = 10, n_dof: int = 6,
                                movable_objects: Tuple[dict, dict] = ({"target object" : [0.0, 0.0, 0.0]}, {}),
                                image_path: str = None, config: dict = None,
                                prompt_file: str = "prompt/dmp_gen_prompt_composer_free.yml",
                                task: str = "N/A"):
        if config is None:
            config = PLACEHOLDER_CONFIG.copy()
        config.update({"dimensions": n_dof, "num_functions": n_functions, "task": task,
                       "movable_objects": self.clean_object_position(movable_objects)})

        prompt_template = self.load_prompt_template(prompt_file)
        system_prompt = self.fill_template(prompt_template["PROMPT_SYSTEM"], config)
        task_prompt = self.fill_template(prompt_template["task_description"], config)
        initial_prompt = self.fill_template(prompt_template["PROMPT_INITIAL"], config)

        content = self.clean_content(task_prompt + "\n\n" + initial_prompt, image_path)
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]

    def generate_followup_message(self, action_plan: str, n_functions: int = 10, n_dof: int = 6,
                                movable_objects: Tuple[dict, dict] = ({"target object" : [0.0, 0.0, 0.0]}, {}),
                                image_path: str = None, config: dict = None,
                                prompt_file: str = "prompt/dmp_gen_prompt_composer_free.yml", 
                                task: str = "N/A"):
        if config is None:
            config = PLACEHOLDER_CONFIG.copy()
        config.update({"dimensions": n_dof, "num_functions": n_functions, "task": task,
                       "movable_objects": self.clean_object_position(movable_objects)})
        
        self.history_corrections[f"trial {len(self.past_weights) - 1}"] = action_plan
        prompt_template = self.load_prompt_template(prompt_file)
        followup_prompt = self.fill_template(prompt_template["PROMPT_FOLLOW_UP"], config)
        followup_prompt = followup_prompt.replace("${.past_weights}", json.dumps(self.past_weights))
        followup_prompt = followup_prompt.replace("${.goal_names}", json.dumps(self.past_goal_names))
        followup_prompt = followup_prompt.replace("${.history}", json.dumps(self.history_corrections))
        action_plan = action_plan or "This is the first trial on this subtask hence there is no past history to report."
        followup_prompt = followup_prompt.replace("${.action_plan}", action_plan)
        followup_prompt = self.clean_content(followup_prompt, image_path)

        return {"role": "user", "content": followup_prompt}

    def get_initial_weights(self, n_functions: int, n_dof: int, movable_objects: Tuple[dict, dict],
                            image_path: str = None, task: str = "N/A"):
        initial_messages = self.generate_initial_message(n_functions, n_dof, movable_objects, image_path, task=task)
        self.messages = initial_messages.copy()

        response = self.client.chat.completions.create(model=GPT_MODEL, temperature=0, messages=initial_messages)
        res_text = response.choices[0].message.content.strip()
        print(colored("[DMP Generator CF] Initial Response: ", "green") + res_text)
        
        weights, angle, height, goal_name, end_gripper_state = self.convert_response_to_weights(res_text)
        self.past_weights["trial 0"] = weights.tolist()        
        self.past_goal_names["trial 0"] = goal_name
        self.messages.append({"role": "assistant", "content": res_text})

        return weights, angle, height, goal_name, end_gripper_state

    def get_followup_weights(self, correction, n_functions, n_dof, movable_objects, image_path=None, task="N/A"):
        followup_message = self.generate_followup_message(correction, n_functions, n_dof, movable_objects, image_path, task=task)
        self.messages.append(followup_message)

        response = self.client.chat.completions.create(model=GPT_MODEL, temperature=0, messages=self.messages)
        res_text = response.choices[0].message.content.strip()
        print(colored("[DMP Generator CF] Follow-up Response: ", "green") + res_text)
        
        weights, angle, height, goal_name, end_gripper_state = self.convert_response_to_weights(res_text)
        self.past_weights[f"trial {len(self.past_weights)}"] = weights.tolist()
        if correction is None:
            self.past_goal_names[f"trial {len(self.past_weights)}"] = goal_name
        self.messages.append({"role": "assistant", "content": res_text})

        return weights, angle, height, goal_name, end_gripper_state

    def convert_response_to_weights(self, response: str):
        start = response.find("{")
        end = response.rfind("}")
        json_response = response[start:end+1]
        weights_dict = json.loads(json_response)
        weights = np.array(weights_dict["weights"])
        angle = float(weights_dict.get("angle", 0.0))
        height = float(weights_dict.get("height", 0.0))
        goal_name = weights_dict.get("goal_name", "")
        end_gripper_state = float(weights_dict.get("end_gripper_state", 0.0))
        return weights, angle, height, goal_name, end_gripper_state

# --- FastAPI Implementation ---
app = FastAPI()
generator = DMPGeneratorCF()
print(colored("[DMP Generator CF] ", "green") + "DMP Generator CF initialized and API server is ready.")

class InitialWeightsRequest(BaseModel):
    n_functions: int
    n_dof: int
    movable_objects: Any
    image_path: Optional[str] = None
    task: str

class FollowupWeightsRequest(BaseModel):
    correction: Optional[str] = None
    n_functions: int
    n_dof: int
    movable_objects: Any
    image_path: Optional[str] = None
    task: str

@app.post("/get_initial_weights")
async def api_get_initial_weights(req: InitialWeightsRequest):
    weights, angle, height, goal_name, end_gripper_state = generator.get_initial_weights(
        n_functions=req.n_functions, n_dof=req.n_dof, movable_objects=req.movable_objects,
        image_path=req.image_path, task=req.task
    )
    return {
        "weights": weights.tolist(), "angle": angle, "height": height,
        "goal_name": goal_name, "end_gripper_state": end_gripper_state,
        "messages": generator.messages
    }

@app.post("/get_followup_weights")
async def api_get_followup_weights(req: FollowupWeightsRequest):
    weights, angle, height, goal_name, end_gripper_state = generator.get_followup_weights(
        correction=req.correction, n_functions=req.n_functions, n_dof=req.n_dof,
        movable_objects=req.movable_objects, image_path=req.image_path, task=req.task
    )
    return {
        "weights": weights.tolist(), "angle": angle, "height": height,
        "goal_name": goal_name, "end_gripper_state": end_gripper_state,
        "messages": generator.messages
    }

@app.post("/refresh")
async def api_refresh():
    global generator
    generator = DMPGeneratorCF()
    return {"status": "success", "message": "DMPGeneratorCF reinitialized"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8031)
