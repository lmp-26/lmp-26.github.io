import os
import cv2
import time
import base64
import argparse
import threading
import numpy as np
from PIL import Image
import pyorbbecsdk as ob 
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from termcolor import colored, cprint
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse

from google import genai
from google.genai import types
from lang_sam import LangSAM

from orbbec_utils import (generate_bbox, parse_response, clean_bbox_dict, VisionDetector, 
                          convert_numpy, get_bbox_from_mask, load_c2r_matrix, 
                          draw_wireframe, draw_uv_indices, show_persistent_plot)

from dotenv import load_dotenv
load_dotenv()

# Conditional imports for DEVA and torch
torch = None
DEVA = None
DEVAInferenceCore = None

track_index = 0
OUTPUT_DIR = Path("./rw_images/output/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETECTION_PROMPT = """
          Point to no more than 10 items in the image. The label returned
          should be an identifying name for the object detected.
          The answer should follow the json format: [{"point": <point>,
          "label": <label1>}, ...]. The points are in [y, x] format
          normalized to 0-1000.
        """

camera_capture: Optional["OrbbecCamera"] = None

class TemporalFilter():
    """Temporal filter for depth frames to reduce noise"""

    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

class OrbbecCamera():
    def __init__(self, output_fps: int, max_frames: int, enable_tracking: bool = False,
                 langsam_device: Optional[str] = None, deva_device: Optional[str] = None):
        self.output_fps = output_fps
        self.current_output_fps = max(1, int(output_fps))
        self.min_output_fps = 5
        self.frame_wait_timeout_ms = 100
        self.fps_backoff_trigger = 6
        self._blocked_wait_count = 0
        self.max_frames = max_frames
        self.min_depth = 200
        self.max_depth = 20000
        self.device_found = False

        # Device configuration
        self.langsam_device = langsam_device
        self.deva_device = deva_device
        
        # If not specified, try to use different GPUs if available
        if self.langsam_device is None or self.deva_device is None:
            import torch as t
            num_gpus = t.cuda.device_count() if t.cuda.is_available() else 0
            if self.langsam_device is None:
                self.langsam_device = "cuda:0"
            if self.deva_device is None:
                self.deva_device = "cuda:1" if num_gpus > 1 else "cuda:0"

        self.langsam_model = LangSAM(device=self.langsam_device) # Orbbec should have a sam model for object segmentation.
        print(colored("[Camera] ", "green") + f"LangSAM model loaded on {self.langsam_device}")

        self.enable_tracking = enable_tracking
        self.tracking_active = False
        self.tracked_object_name = None
        self.tracked_mask = None
        self.tracked_center = None
        self.tracked_bbox = None
        self.tracking_thread = None
        self.deva_core = None
        self.deva_model = None
        
        if self.enable_tracking:
            self._load_deva_modules()

    def _load_deva_modules(self):
        global torch, DEVA, DEVAInferenceCore
        if torch is None:
            print(colored("[Camera] ", "green") + "Loading torch and DEVA modules...")
            import torch as t
            torch = t
            
            import sys
            deva_path = os.path.join(os.path.dirname(__file__), "Tracking-Anything-with-DEVA")
            if deva_path not in sys.path:
                sys.path.append(deva_path)
            
            from deva.model.network import DEVA as DEVA_Model
            from deva.inference.inference_core import DEVAInferenceCore as DIC
            DEVA = DEVA_Model
            DEVAInferenceCore = DIC
            print(colored("[Camera] ", "green") + "DEVA modules loaded.")

    def _init_deva(self):
        if self.deva_model is None:
            self._load_deva_modules()
            print(colored("[Camera] ", "green") + "Initializing DEVA model weights...")
            checkpoint_path = os.path.join(os.path.dirname(__file__), "Tracking-Anything-with-DEVA/saves/DEVA-propagation.pth")
            cfg = {
                'model': checkpoint_path,
                'key_dim': 64,
                'value_dim': 512,
                'pix_feat_dim': 512,
                'disable_long_term': True,
                'max_mid_term_frames': 3,
                'min_mid_term_frames': 1,
                'max_long_term_elements': 128,
                'num_prototypes': 32,
                'top_k': 50,
                'mem_every':30,
                'chunk_size': 1,
                'size': 256,
                'enable_long_term': False,
                'enable_long_term_count_usage': False,
                'max_missed_detection_count': 20
            }
            self.deva_model = DEVA(cfg).to(self.deva_device).eval()
            model_weights = torch.load(checkpoint_path, map_location=self.deva_device)
            self.deva_model.load_weights(model_weights)
            self.deva_cfg = cfg
            del model_weights
            torch.cuda.empty_cache()
            print(colored("[Camera] ", "green") + f"DEVA model initialized on {self.deva_device} with size 360.")

    def check_for_device(self):
        self.device_found = False
        context = ob.Context()
        device_list = context.query_devices()
        if device_list.get_count() >= 1:
            self.device_found = True
        else:
            cprint("[Camera] Error: No Orbbec camera found.", "red")
        return self.device_found
    
    def get_stream_config(self):
        """
        Get matched color and depth stream profiles
        """
        try:
            profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            assert profile_list is not None
            
            for i in range(len(profile_list)):
                color_profile = profile_list[i]

                if color_profile.get_format() != ob.OBFormat.RGB:
                    continue

                d2c_profile_list = self.pipeline.get_d2c_depth_profile_list(color_profile, ob.OBAlignMode.HW_MODE)
                if len(d2c_profile_list) == 0:
                    continue

                d2c_profile = d2c_profile_list[0]
                print(colored("[Camera] ", "green") + f"Found compatible d2c color profile: {color_profile}, depth profile: {d2c_profile}")

                self.config.enable_stream(d2c_profile)
                self.config.enable_stream(color_profile)

                self.config.set_align_mode(ob.OBAlignMode.HW_MODE)
                return self.config
        except Exception as e:
            cprint("[Camera] Error while getting stream config: " + str(e), "red")
            return None
    
    def start(self):
        self.pipeline = ob.Pipeline()
        self.config = ob.Config()
        self.color_buffer = deque(maxlen=self.max_frames)
        self.depth_buffer = deque(maxlen=self.max_frames)
        self.aligned_buffer = deque(maxlen=self.max_frames)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.temporal_filter = TemporalFilter(alpha=0.1)
        self.point_cloud_filter = ob.PointCloudFilter()

        # set stream config
        self.get_stream_config()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if hasattr(self, "pipeline"):
            self.pipeline.stop()

    def _capture_loop(self):
        self.pipeline.enable_frame_sync()
        self.pipeline.start(self.config)
        param = self.pipeline.get_camera_param()
        print(colored("[Camera] ", "green") + f"Camera parameters: {param.depth_intrinsic}")
        self.fx, self.fy, self.cx, self.cy = param.depth_intrinsic.fx, param.depth_intrinsic.fy, param.depth_intrinsic.cx, param.depth_intrinsic.cy
        self.color_fx, self.color_fy, self.color_cx, self.color_cy = param.rgb_intrinsic.fx, param.rgb_intrinsic.fy, param.rgb_intrinsic.cx, param.rgb_intrinsic.cy
        dist = param.rgb_distortion
        self.color_distortion = [dist.k1, dist.k2, dist.p1, dist.p2, dist.k3, dist.k4, dist.k5, dist.k6]
        frame_counter = 0
        last_time = time.time_ns()
        while self.running:
            wait_start = time.time()
            frames = self.pipeline.wait_for_frames(self.frame_wait_timeout_ms)
            wait_elapsed = time.time() - wait_start

            target_interval = 1.0 / max(1, self.current_output_fps)
            if frames is None or wait_elapsed > target_interval * 1.5:
                self._blocked_wait_count += 1
            else:
                self._blocked_wait_count = max(0, self._blocked_wait_count - 1)

            if self._blocked_wait_count >= self.fps_backoff_trigger and self.current_output_fps > self.min_output_fps:
                self.current_output_fps = max(self.min_output_fps, self.current_output_fps - 2)
                self._blocked_wait_count = 0
                cprint(f"[Camera] Stream blocking detected. Lowering output fps to {self.current_output_fps}.", "yellow")

            if frames is None:
                continue

            curr_time = time.time_ns()
            remaining_time = (1 / max(1, self.current_output_fps)) - (curr_time - last_time) / 1e9
            if (0.0001 < remaining_time < 0.2):
                time.sleep(remaining_time)
            # print((time.time_ns() - last_time) / 1e9)
            last_time = time.time_ns()

            frame_counter += 1

            # Get RGB Frame
            rgb_frame = frames.get_color_frame()
            # Get Depth Frame
            depth_frame = frames.get_depth_frame()
            if rgb_frame is None or depth_frame is None:
                continue

            width, height = rgb_frame.get_width(), rgb_frame.get_height()
            assert rgb_frame.get_format() == ob.OBFormat.RGB
            rgb_image = np.resize(np.asanyarray(rgb_frame.get_data()), (height, width, 3))
            rgb_image = cv2.flip(rgb_image, 0)
            rgb_image = cv2.flip(rgb_image, 1)

            with self.lock:
                timestamp = datetime.now()
                self.color_buffer.append({
                    "frame": rgb_image,
                    "timestamp": timestamp
                })

            width, height = depth_frame.get_width(), depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale
            depth_data = np.where((depth_data > self.min_depth) & (depth_data < self.max_depth), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)
            depth_data = self.temporal_filter.process(depth_data)
            depth_data = cv2.flip(depth_data, 0)
            depth_data = cv2.flip(depth_data, 1)

            with self.lock:
                timestamp = datetime.now()
                self.depth_buffer.append({
                    "frame": depth_data,
                    "timestamp": timestamp
                })

    def get_latest_frames(self):
        latest_color = None
        latest_depth = None
        if len(self.color_buffer) > 0 and len(self.depth_buffer) > 0:
            with self.lock:
                latest_color = self.color_buffer[-1]
                latest_depth = self.depth_buffer[-1]
        return latest_color, latest_depth
    
    def get_image(self):
        return self.get_latest_frames()[0]
    
    def object_detection(self, image=None):
        """Run object detection on an in-memory frame payload from self.color_buffer."""
        if image is None:
            image = self.get_image()
        assert image is not None, "No image available for object detection."

        if isinstance(image, dict):
            if "frame" not in image:
                raise ValueError("Image dict must include a 'frame' key.")
            frame = image["frame"]
        else:
            frame = image
        assert isinstance(frame, np.ndarray), "Input image must be a numpy array."

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".png", frame_bgr)
        assert success, "Failed to encode image for object detection."
        image_bytes = encoded.tobytes()

        api_key = os.environ.get("GENAI_API_KEY", "GET_YOUR_OWN_KEY")
        client = genai.Client(api_key = api_key)
        image_response = client.models.generate_content(
            model="gemini-robotics-er-1.6-preview",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                ),
                DETECTION_PROMPT
            ],
            config = types.GenerateContentConfig(
                temperature=0.5,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        client.close()
        del client

        print(colored("[Camera] ", "green") + f"Object detection response: {image_response.text}")
        return image_response.text

    def find_objects(self, query: str = ""):
        print(colored("[Camera] ", "green") + "Running object detection...")
        last_frame, last_depth = self.get_latest_frames()
        assert last_frame is not None and last_depth is not None, "No frames available for object detection."

        with torch.no_grad():
            if query == "":
                llm_response = self.object_detection(last_frame)
                assert llm_response is not None and isinstance(llm_response, str), "LLM did not return a valid response for object detection."
                json_response = parse_response(llm_response)
                assert json_response is not None and isinstance(json_response, list), "Parsed LLM response is not a valid list of detections."
            else:
                json_response = [{"label": query, "point": [0, 0]}]

            last_frame_bgr = cv2.cvtColor(last_frame["frame"], cv2.COLOR_RGB2BGR)
            obbs, annotated_bgr = generate_bbox(depth_mm=last_depth["frame"], 
                                                color_bgr=last_frame_bgr, 
                                                detections=json_response,
                                                langsam_model=self.langsam_model,
                                                fx=self.color_fx, fy=self.color_fy, cx=self.color_cx, cy=self.color_cy,
                                                query=query)
        
        # show_persistent_plot("Object Detection", cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
        success, buffer = cv2.imencode(".png", annotated_bgr)
        assert success, "Failed to encode annotated image for visualization."

        bbox = clean_bbox_dict(obbs)
        center = {item["label"].lower(): item.get("centers", None) for item in obbs}
        masks = {item["label"].lower(): item.get("mask", None) for item in obbs}
        content = {
            "image": base64.b64encode(buffer.tobytes()).decode("utf-8"),
            "objects": center,
            "bbox": bbox,
            "masks": masks,
        }
        
        torch.cuda.empty_cache()
        return content

    def read_ArUco(self, image: Optional[np.ndarray] = None):
        if image is None:
            image = self.get_image()
        assert image is not None, "No image available for object detection."

        if isinstance(image, dict):
            if "frame" not in image:
                raise ValueError("Image dict must include a 'frame' key.")
            frame = image["frame"]
        else:
            frame = image
        assert isinstance(frame, np.ndarray), "Input image must be a numpy array."
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        aruco_detector = VisionDetector(
                                        fx=self.color_fx, fy=self.color_fy, cx=self.color_cx, cy=self.color_cy, 
                                        distortion=self.color_distortion,
                                        marker_size=0.06)
        # calibration
        # Ry = np.array([
        #         [ np.cos(-np.pi/2), 0, np.sin(-np.pi/2)],
        #         [ 0,                 1, 0               ],
        #         [-np.sin(-np.pi/2), 0, np.cos(-np.pi/2)]], dtype=np.float64)
        # Rx = np.array([
        #     [1, 0, 0],
        #     [0, np.cos(np.pi), -np.sin(np.pi)],
        #     [0, np.sin(np.pi),  np.cos(np.pi)]
        # ])
        Ry = np.array([
                [ np.cos(np.pi), 0, np.sin(np.pi)],
                [ 0,                 1, 0               ],
                [-np.sin(np.pi), 0, np.cos(np.pi)]], dtype=np.float64)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(-np.pi/2), -np.sin(-np.pi/2)],
            [0, np.sin(-np.pi/2),  np.cos(-np.pi/2)]
        ])
        R = Ry @ Rx
        rvecs, tvecs, corners, ids, frame_markers = aruco_detector.plot_aruco(frame_bgr, rotation_matrix=R)
        marker_poses_camera, marker_poses_robot = aruco_detector.pose_vectors_to_cart(rvecs, tvecs, ids=ids)
        print(colored("[Camera] ", "green") + f"Detected ArUco markers with IDs: {ids.flatten() if ids is not None else 'None'}")
        print(colored("[Camera] ", "green") + f"Marker Cartesian poses in camera frame: {marker_poses_camera}")
        print(colored("[Camera] ", "green") + f"Marker Cartesian poses in robot frame: {marker_poses_robot}")
        show_persistent_plot("AruCo", cv2.cvtColor(frame_markers, cv2.COLOR_BGR2RGB))
        return marker_poses_camera, marker_poses_robot

    def start_tracking(self, obj_name: str):
        if not self.enable_tracking:
            raise ValueError("Tracking is not enabled for this camera instance.")
        
        if self.tracking_active:
            print(colored("[Camera] ", "yellow") + f"Stopping existing tracking for {self.tracked_object_name}")
            self.tracking_active = False
            if self.tracking_thread:
                self.tracking_thread.join()

        self.tracked_object_name = obj_name
        self.tracking_active = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        print(colored("[Camera] ", "green") + f"Started tracking for: {obj_name}")

    def stop_tracking(self):
        self.tracking_active = False
        if self.tracking_thread:
            self.tracking_thread.join()
        print(colored("[Camera] ", "green") + "Tracking stopped.")

    def _tracking_loop(self):
        try:
            self._init_deva()
            self.deva_core = DEVAInferenceCore(self.deva_model, config=self.deva_cfg)
            # self.deva_core.enabled_long_id()
            
            first_frame = True
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            C2R_np = load_c2r_matrix()
            if C2R_np is None:
                print(colored("[Camera] ", "red") + "C2R.npy not found or invalid. Aborting tracking.")
                self.tracking_active = False
                return

            frame_count = 0

            proc_h = 360
            proc_w = None

            with torch.no_grad():
                while self.tracking_active:
                    latest_color, latest_depth = self.get_latest_frames()
                    if latest_color is None or latest_depth is None:
                        time.sleep(0.01)
                        continue

                    frame_rgb = latest_color["frame"]
                    depth_mm = latest_depth["frame"]

                    if proc_w is None:
                        h, w = frame_rgb.shape[:2]
                        proc_w = int(w * proc_h / h)
                        # Scale intrinsics for the processing resolution
                        scale_x = proc_w / w
                        scale_y = proc_h / h
                        proc_fx, proc_fy = self.color_fx * scale_x, self.color_fy * scale_y
                        proc_cx, proc_cy = self.color_cx * scale_x, self.color_cy * scale_y
                    
                    # Resize color frame once for both DEVA and BBox processing
                    frame_proc = cv2.resize(frame_rgb, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
                    # Also resize depth to match mask
                    depth_proc = cv2.resize(depth_mm, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)

                    # DEVA expects torch tensor 3*H*W
                    frame_torch = torch.from_numpy(frame_proc.copy()).permute(2, 0, 1).float().to(self.deva_device)
                    frame_torch.div_(255.0)

                    if first_frame:
                        print(colored("[Camera] ", "green") + f"Initializing tracking mask for {self.tracked_object_name} using LangSAM on {self.langsam_device}...")
                        image_pil = Image.fromarray(frame_rgb)
                        try:
                            pred = self.langsam_model.predict([image_pil], [self.tracked_object_name], box_threshold=0.3)
                            if not pred or len(pred[0]["masks"]) == 0:
                                print(colored("[Camera] ", "red") + f"LangSAM failed to find {self.tracked_object_name}. Aborting tracking.")
                                self.tracking_active = False
                                break
                            
                            print(colored("[Camera] ", "green") + f"LangSAM found {self.tracked_object_name}. Initializing DEVA.")
                            # Use the first mask found
                            mask_np = (np.asarray(pred[0]["masks"][0]) > 0).astype(np.uint8)
                            # Resize mask to match frame_proc
                            mask_torch = torch.from_numpy(cv2.resize(mask_np, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)).float().to(self.deva_device)
                            
                            # DEVA step with initial mask
                            self.deva_core.step(frame_torch, mask_torch, objects=[1])
                            
                            del mask_torch, mask_np, pred, image_pil
                            torch.cuda.empty_cache()
                            first_frame = False
                        except Exception as e:
                            print(colored("[Camera] ", "red") + f"Error in LangSAM initialization: {e}")
                            self.tracking_active = False
                            break
                    else:
                        # DEVA propagation step
                        prob = self.deva_core.step(frame_torch)
                        pred_mask_torch = torch.argmax(prob, dim=0)
                        pred_mask = pred_mask_torch.cpu().numpy().astype(np.uint8)
                        
                        del prob, pred_mask_torch
                        
                        num_pixels = np.sum(pred_mask == 1)
                        if num_pixels < 10:
                            print(colored("[Camera Tracking] ", "yellow") + f"LOST: {self.tracked_object_name} (mask pixels: {num_pixels})")
                        
                        final_mask_proc = (pred_mask == 1).astype(np.uint8)
                        self.tracked_mask = final_mask_proc

                        annotated_proc = frame_proc.copy()
                        res = get_bbox_from_mask(final_mask_proc, depth_proc, annotated_proc, kernel, 
                                                 proc_fx, proc_fy, proc_cx, proc_cy, C2R_np)
                        
                        if res:
                            corners_xyz, corners_uv, corners_pxpyz, center_robot_np = res
                            self.tracked_center = center_robot_np
                            print(colored("[Camera Tracking] ", "cyan") + f"FOUND: {self.tracked_object_name} at {center_robot_np}")
                            
                            obbs_item = [{"label": self.tracked_object_name, "corners": corners_xyz, "pixel_corners": corners_pxpyz}]
                            bbox_dict = clean_bbox_dict(obbs_item)
                            if self.tracked_object_name in bbox_dict:
                                self.tracked_bbox = bbox_dict[self.tracked_object_name]
                            
                            draw_wireframe(annotated_proc, corners_uv, self.tracked_object_name)
                            draw_uv_indices(annotated_proc, corners_uv)
                            show_persistent_plot("Tracking", annotated_proc)
                        else:
                            self.tracked_center = None
                            self.tracked_bbox = None
                            if num_pixels >= 10:
                                print(colored("[Camera Tracking] ", "red") + f"BBOX FAIL: {self.tracked_object_name} has {num_pixels} pixels but no 3D pos.")
                        
                        del final_mask_proc, pred_mask
                    
                    del frame_torch
                    
                    frame_count += 1
                    if frame_count % 50 == 0:
                        torch.cuda.empty_cache()
                        
                    # Run as fast as possible, but yield a bit
                    time.sleep(0.001)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(colored("[Camera] ", "red") + f"Tracking loop error: {e}")
            self.tracking_active = False



@asynccontextmanager
async def lifespan(_app: FastAPI):
    global camera_capture
    camera_capture = OrbbecCamera(
        output_fps=20,
        max_frames=20 * 60, # buffer for 60 seconds
        enable_tracking=True # Default to true for FastAPI
    )
    if not camera_capture.check_for_device():
        return
    
    camera_capture.start()
    yield
    camera_capture.stop()




def get_frames_for_duration(cam: "OrbbecCamera", duration_seconds: float):
    with cam.lock:
        frames = list(cam.color_buffer)
        if not frames: return []
        end_ts = frames[-1]["timestamp"]
        start_ts = end_ts - timedelta(seconds=duration_seconds)
        return [f for f in frames if f["timestamp"] >= start_ts]

def create_video(frames: list, output_fps: int, output_filename: str) -> Path:
    output_path = OUTPUT_DIR / output_filename
    if not frames: return output_path
    height, width, _ = frames[0]['frame'].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
    for frame_info in frames:
        # Convert RGB back to BGR for OpenCV VideoWriter
        bgr_frame = cv2.cvtColor(frame_info['frame'], cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)
    video_writer.release()
    return output_path


def main():
    global camera_capture
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--FastAPI",
        action="store_true",
        help="use FastAPI server to serve camera data",
    )
    parser.add_argument(
        "--enable_tracking",
        action="store_true",
        help="Enable object tracking with DEVA",
    )
    parser.add_argument(
        "--langsam_device",
        type=str,
        default=None,
        help="Device to run LangSAM on (e.g., cuda:0, cuda:1)",
    )
    parser.add_argument(
        "--deva_device",
        type=str,
        default=None,
        help="Device to run DEVA on (e.g., cuda:0, cuda:1)",
    )
    args = parser.parse_args()

    if args.FastAPI:
        import uvicorn
        app = FastAPI(lifespan=lifespan)

        @app.get("/health")
        async def health_check():
            return {"status": "healthy"}

        @app.get("/latest")
        async def get_latest():
            cam = camera_capture
            if not cam: raise HTTPException(status_code=503, detail="Camera not initialized")
            color, depth = cam.get_latest_frames()
            if not color or not depth: raise HTTPException(status_code=503, detail="No frames available")
            _, c_buf = cv2.imencode('.png', cv2.cvtColor(color['frame'], cv2.COLOR_RGB2BGR))
            _, d_buf = cv2.imencode('.png', depth['frame'])
            return JSONResponse(content={
                "color": base64.b64encode(c_buf).decode('utf-8'),
                "depth": base64.b64encode(d_buf).decode('utf-8'),
                "timestamp": color['timestamp'].isoformat(),
                "format": "png"
            })

        @app.get("/find_objects")
        async def find_objects(query: str = ""):
            cam = camera_capture
            if cam is None:
                raise HTTPException(status_code=503, detail="Camera is not initialized")
            content = cam.find_objects(query)
            return JSONResponse(content = convert_numpy(content))
        
        @app.get("/start_tracking")
        async def start_tracking(query: str):
            cam = camera_capture
            if cam is None:
                raise HTTPException(status_code=503, detail="Camera is not initialized")
            try:
                cam.start_tracking(query)
                return {"status": f"Started tracking {query}"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/stop_tracking")
        async def stop_tracking():
            cam = camera_capture
            if cam is None:
                raise HTTPException(status_code=503, detail="Camera is not initialized")
            cam.stop_tracking()
            return {"status": "Stopped tracking"}

        @app.get("/tracked_object")
        async def get_tracked_object():
            cam = camera_capture
            if cam is None:
                raise HTTPException(status_code=503, detail="Camera is not initialized")
            if not cam.tracking_active:
                return {"status": "Not tracking any object"}
            
            return JSONResponse(content=convert_numpy({
                "label": cam.tracked_object_name,
                "center": cam.tracked_center,
                "bbox": cam.tracked_bbox
            }))

        @app.get("/latest/color")
        async def get_latest_color_image():
            """Get the most recent color PNG image"""
            cam = camera_capture
            if cam is None:
                raise HTTPException(status_code=503, detail="Camera is not initialized")

            latest_frame, latest_depth = cam.get_latest_frames()

            if latest_frame is None:
                error_msg = f"No color frames available yet."
                print(f"[503] {error_msg}")
                raise HTTPException(status_code=503, detail=error_msg)

            # Encode frame to PNG in memory
            latest_frame_bgr = cv2.cvtColor(latest_frame['frame'], cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode('.png', latest_frame_bgr)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to encode image")

            return Response(
                content=buffer.tobytes(),
                media_type="image/png")

        @app.get("/latest/depth")
        async def get_latest_depth_image():
            cam = camera_capture
            if not cam: raise HTTPException(status_code=503, detail="Camera not initialized")
            _, depth = cam.get_latest_frames()
            if not depth: raise HTTPException(status_code=503, detail="No depth frames")
            _, buffer = cv2.imencode('.png', depth['frame'])
            return Response(content=buffer.tobytes(), media_type="image/png")

        @app.get("/video/{duration_seconds}")
        async def get_video(duration_seconds: float):
            cam = camera_capture
            if not cam: raise HTTPException(status_code=503, detail="Camera not initialized")
            frames = get_frames_for_duration(cam, duration_seconds)
            if not frames: raise HTTPException(status_code=503, detail="No frames available")
            output_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = create_video(frames, cam.output_fps, output_filename)
            return FileResponse(output_path, media_type="video/mp4", filename=output_filename)

        @app.get("/track_object")
        async def track_object():
            cam = camera_capture
            if cam is None:
                raise HTTPException(status_code=503, detail="Camera is not initialized")
            
            if not cam.tracking_active or cam.tracked_center is None:
                # Return 200 with a status instead of 404 to reduce log noise
                return JSONResponse(content={"status": "not_found", "position": None})
            
            # Use the real tracked center from the tracking loop
            # cam.tracked_center is [x, y, z] in robot frame
            # Return format expected by robot: [x, y, z, yaw, gripper]
            try:
                # Ensure it's a list
                current_center = list(cam.tracked_center)
                goal = current_center + [0.0, 1.0]
                return JSONResponse(content={"position": goal})
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing tracked center: {str(e)}")

        uvicorn.run(app, host="0.0.0.0", port=8010)

    else: 
        camera_capture = OrbbecCamera(
            output_fps=20,
            max_frames=20 * 60, # buffer for 60 seconds
            enable_tracking=args.enable_tracking,
            langsam_device=args.langsam_device,
            deva_device=args.deva_device
        )
        camera_capture.start()
        time.sleep(8) # allow camera to warm up and fill buffer
        while True:
            try:
                # test = input("****************************** ")
                camera_capture.read_ArUco()
                continue
            except KeyboardInterrupt:
                print("Exiting...")
                break


if __name__ == "__main__":
    main()
