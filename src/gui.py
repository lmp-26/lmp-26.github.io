import asyncio
import base64
import html
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from nicegui import app, ui

# --- Configuration ---
HEALTH_CHECK_INTERVAL = 5
CAMERA_UPDATE_INTERVAL = 5
UI_UPDATE_INTERVAL = 0.1

CACHE_DIR = Path(__file__).resolve().parent / "lmp" / "cache"
app.add_static_files('/cache', str(CACHE_DIR))

API_ENDPOINTS = {
    "Orchestrator": "http://localhost:7999",
    "Camera": "http://localhost:8010",
    "Controller": "http://localhost:8030",
    "LLM": "http://localhost:8031",
}

# --- State Management ---
def init_state():
    """Initialize app storage and global state objects."""
    if 'log_messages' not in app.storage.general:
        app.storage.general.update({
            'log_messages': [],
            'current_task': 'None',
            'yolo_image_data': None,
            'dmp_plot_data': None,
            'dmp_carousel_images': [],
            'dmp_carousel_index': 0,
            'task_video_path': None,
            'confirmation_request': None,
            'prompt_text': None,
            'prompt_dirty': False,
            'prompt_messages': None,
        })
    if not hasattr(app, '_confirmation_futures'):
        app._confirmation_futures = {}

init_state()
ui_state_lock = asyncio.Lock()

# --- Helpers ---
def render_prompt_messages_html(messages: List[Dict]) -> str:
    """Renders colored HTML for prompt messages."""
    role_color = {
        "system": "#1e88e5",     # blue
        "user": "#90caf9",       # light blue
        "assistant": "#81c784",  # light green
    }
    blocks = []
    for m in messages:
        role = str(m.get("role", "unknown"))
        content = m.get("content", "")
        pretty = json.dumps(content, ensure_ascii=False, indent=2) if isinstance(content, (dict, list)) else str(content)
        pretty = html.escape(pretty).replace("\n", "<br>")
        color = role_color.get(role, "#bdbdbd")
        blocks.append(f"""
            <div style="margin:0; padding:8px 10px; border-bottom:1px solid #333;">
              <div style="font-weight:700; color:{color}; margin-bottom:4px; text-transform: uppercase; font-size: 10px;">{html.escape(role)}</div>
              <div style="font-family:ui-monospace, monospace; font-size:12px; line-height:1.4; color: #eee;">{pretty}</div>
            </div>
        """)
    return "".join(blocks)

# --- Components ---
class HealthMonitor:
    def __init__(self):
        self.status_badges: Dict[str, ui.badge] = {}
        self.statuses: Dict[str, bool] = {name: False for name in API_ENDPOINTS}

    async def update_all_health(self):
        async with httpx.AsyncClient(timeout=2.0) as client:
            for name, url in API_ENDPOINTS.items():
                try:
                    response = await client.get(f"{url}/health")
                    self.statuses[name] = response.status_code == 200
                except Exception:
                    self.statuses[name] = False
                self._update_badge(name)

    def _update_badge(self, name: str):
        if name in self.status_badges:
            active = self.statuses[name]
            self.status_badges[name].props(f'color={"green" if active else "red"}')
            self.status_badges[name].set_text(f"{name}: {'Online' if active else 'Offline'}")

    def create_badge(self, name: str):
        badge = ui.badge(f"{name}: Offline", color='red').props('outline')
        self.status_badges[name] = badge
        return badge

class RobotDashboard:
    def __init__(self):
        self.task_input = None
        self.task_log = None
        self.task_log_scroll = None
        self.prompt_log = None
        self.prompt_log_scroll = None
        self.camera_image = None
        self.yolo_image = None
        self.dmp_plot_image = None
        self.task_video = None
        self.dmp_counter = None
        self.dmp_prev_btn = None
        self.dmp_next_btn = None
        self.task_progress_label = None
        self.dynamic_buttons_container = None
        self.approval_future = None
        self.simulation_title_label = None

    def render(self):
        ui.dark_mode().enable()
        
        # Header
        with ui.header(elevated=True).classes('bg-grey-10 items-center justify-between q-pa-sm'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('smart_toy', size='md').classes('text-primary')
                ui.label('LMP DASHBOARD').classes('text-h6 text-weight-bold tracking-tighter')
            
            with ui.row().classes('items-center gap-2'):
                for name in API_ENDPOINTS:
                    health_monitor.create_badge(name)

        # Left Sidebar (Drawer)
        with ui.left_drawer(value=True, elevated=True).classes('bg-grey-9 q-pa-md flex column gap-4').props('width=300'):
            self._render_controls()
            self._render_progress_section()
            self._render_logs()

        # Main Area
        with ui.column().classes('w-full max-w-6xl mx-auto q-pa-md gap-6'):
            self._render_media_feeds()

    def _render_controls(self):
        with ui.card().classes('w-full bg-grey-8 shadow-2'):
            ui.label('CONTROLS').classes('text-caption text-weight-bold text-grey-5')
            self.task_input = ui.input(label='Task Command', placeholder='e.g. Put banana on plate').classes('w-full')
            with ui.row().classes('w-full items-center gap-1 mt-2 no-wrap'):
                ui.button('EXECUTE', on_click=lambda: self.execute_task(self.task_input.value)).props('unelevated rounded color=primary icon=play_arrow').classes('flex-grow text-weight-bold')
                ui.button(on_click=self.home_robot).props('flat round icon=home color=grey-4').tooltip('Home Robot')
                ui.button(on_click=self.return_to_previous).props('flat round icon=undo color=grey-4').tooltip('Return to Previous')

    def _render_progress_section(self):
        with ui.card().classes('w-full bg-grey-8 shadow-2'):
            ui.label('STATUS').classes('text-caption text-weight-bold text-grey-5')
            self.task_progress_label = ui.label('Idle').classes('text-body1 text-weight-medium text-primary')
            self.dynamic_buttons_container = ui.row().classes('w-full gap-2 mt-2')

    def _render_logs(self):
        # Task Log
        with ui.card().classes('w-full bg-grey-8 shadow-2 flex-grow overflow-hidden'):
            ui.label('TASK LOG').classes('text-caption text-weight-bold text-grey-5')
            self.task_log_scroll = ui.scroll_area().classes('w-full flex-grow h-48 border border-grey-7 rounded-sm mt-1 bg-grey-10')
            with self.task_log_scroll:
                self.task_log = ui.column().classes('w-full q-pa-xs').style('gap: 2px; font-size: 11px; font-family: monospace;')

        # Prompt Log
        with ui.card().classes('w-full bg-grey-8 shadow-2 flex-grow overflow-hidden'):
            ui.label('PROMPT LOG').classes('text-caption text-weight-bold text-grey-5')
            self.prompt_log_scroll = ui.scroll_area().classes('w-full flex-grow h-64 border border-grey-7 rounded-sm mt-1 bg-grey-10')
            with self.prompt_log_scroll:
                self.prompt_log = ui.column().classes('w-full q-pa-none').style('gap: 0;')

    def _render_media_feeds(self):
        with ui.grid(columns=2).classes('w-full gap-4'):
            # Camera
            with ui.card().classes('bg-grey-9 border border-grey-8 overflow-hidden'):
                ui.label('LIVE CAMERA').classes('text-subtitle2 text-weight-bold text-grey-4 q-pa-sm')
                self.camera_image = ui.image().classes('w-full bg-black').style('height: 300px; object-fit: contain;')
            
            # YOLO
            with ui.card().classes('bg-grey-9 border border-grey-8 overflow-hidden'):
                ui.label('OBJECT DETECTION').classes('text-subtitle2 text-weight-bold text-grey-4 q-pa-sm')
                self.yolo_image = ui.image().classes('w-full bg-black').style('height: 300px; object-fit: contain;')

            # Controller Simulation (Full Width)
            with ui.card().classes('col-span-full bg-grey-9 border border-grey-8 overflow-hidden'):
                with ui.row().classes('w-full items-center justify-between q-pa-sm'):
                    self.simulation_title_label = ui.label('DMP TRAJECTORY SIMULATION').classes('text-subtitle2 text-weight-bold text-grey-4')
                    with ui.row().classes('items-center gap-2'):
                        self.dmp_prev_btn = ui.button(icon='chevron_left', on_click=self.dmp_carousel_prev).props('flat dense').set_enabled(False)
                        self.dmp_counter = ui.label('0 / 0').classes('text-caption text-grey-5')
                        self.dmp_next_btn = ui.button(icon='chevron_right', on_click=self.dmp_carousel_next).props('flat dense').set_enabled(False)
                
                with ui.element('div').classes('relative w-full'):
                    self.dmp_plot_image = ui.image().classes('w-full bg-black shadow-4').style('min-height: 400px; max-height: 80vh; object-fit: contain;')
                    self.task_video = ui.video('/cache/task_video.mp4').classes('w-full bg-black shadow-4').style('min-height: 400px; max-height: 80vh;')
                    self.task_video.props('autoplay muted loop playsinline controls')
                    self.task_video.set_visibility(False)
                    
                    with ui.column().classes('absolute-full items-center justify-center z-10') \
                        .style('background: rgba(0, 0, 0, 0.7); backdrop-filter: blur(8px); transition: all 0.3s;') \
                        as self.rollout_overlay:
                        ui.icon('hourglass_empty', size='xl').classes('text-grey-5 mb-2')
                        ui.label('Last rollout not available yet').classes('text-h6 text-grey-4 text-weight-medium')
                    self.rollout_overlay.set_visibility(False)

    # --- Actions ---
    async def execute_task(self, command: str):
        if not command: return
        self.task_log.clear()
        
        # Show overlay at start of task
        if self.rollout_overlay:
            self.rollout_overlay.set_visibility(True)

        async with ui_state_lock:
            app.storage.general.update({'log_messages': [], 'dmp_carousel_images': [], 'dmp_carousel_index': 0})
        self.update_dmp_carousel()
        
        app.storage.general['current_task'] = command
        self._update_task_progress_label()
        self._add_log_entry(f"🚀 Starting task: {command}")
        
        asyncio.create_task(self._run_task_api_call(command))

    async def _run_task_api_call(self, command: str):
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(f"{API_ENDPOINTS['Orchestrator']}/task", json={"task": command})
                response.raise_for_status()
                success = response.json().get("success", False)
                self._add_log_entry("✅ Task completed successfully!" if success else "❌ Task failed.")
        except Exception as e:
            self._add_log_entry(f"⚠️ Error: {e}")
        finally:
            app.storage.general['current_task'] = "None"
            self._update_task_progress_label()
            async with ui_state_lock:
                app.storage.general.update({'dmp_carousel_images': [], 'dmp_carousel_index': 0})
            self.update_dmp_carousel()

    def _add_log_entry(self, text: str):
        with self.task_log:
            ui.html(f'<div class="text-grey-3">{text}</div>')
        self.task_log_scroll.scroll_to(percent=1.0)

    def _update_task_progress_label(self):
        if self.task_progress_label:
            task = app.storage.general.get('current_task', 'None')
            self.task_progress_label.set_text(task if task else 'Idle')

    async def wait_for_option_selection(self, options: List[str]):
        # Clear plots to show rollout video ONLY for post-rollout judgment
        # (if 'Success' or 'Complete' are in the options)
        if any(opt in ['Success', 'Complete', 'Failure'] for opt in options):
            async with ui_state_lock:
                app.storage.general.update({'dmp_carousel_images': [], 'dmp_carousel_index': 0})
            self.update_dmp_carousel()
            # Hide overlay after rollout finishes
            if self.rollout_overlay:
                self.rollout_overlay.set_visibility(False)

        self.approval_future = asyncio.Future()
        self.clear_dynamic_buttons()

        with self.dynamic_buttons_container:
            button_configs = {
                'Success': {'color': 'green', 'icon': 'check_circle'},
                'Failure': {'color': 'red', 'icon': 'cancel'},
                'Feedback': {'color': 'orange', 'icon': 'rate_review'},
                'Complete': {'color': 'blue', 'icon': 'task_alt'},
                'Abort': {'color': 'deep-orange', 'icon': 'stop_circle'},
                'Continue': {'color': 'primary', 'icon': 'arrow_forward_ios'},
            }
            for opt in options:
                config = button_configs.get(opt, {'color': 'primary', 'icon': None})
                if opt == 'Feedback':
                    ui.button(opt, on_click=self._show_feedback_dialog).props(f'unelevated rounded color={config["color"]} icon={config["icon"]}').classes('text-weight-bold')
                else:
                    ui.button(opt, on_click=lambda o=opt: self._on_option_selected(o)).props(f'unelevated rounded color={config["color"]} icon={config["icon"]}').classes('text-weight-bold')
        
        result = await self.approval_future
        self.clear_dynamic_buttons()
        return result

    def _on_option_selected(self, option: str):
        if option in ['Abort', 'Complete'] and self.rollout_overlay:
            self.rollout_overlay.set_visibility(True)
        if self.approval_future and not self.approval_future.done():
            self.approval_future.set_result({"choice": option, "feedback": None})

    def _show_feedback_dialog(self):
        with ui.dialog() as dialog, ui.card().classes('w-96'):
            ui.label('PROVIDE FEEDBACK').classes('text-h6 text-primary')
            feedback_input = ui.textarea(label='Details', placeholder='What should be improved?').classes('w-full h-32')
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('CANCEL', on_click=dialog.close).props('flat')
                ui.button('SUBMIT', on_click=lambda: self._submit_feedback(dialog, feedback_input.value)).props('color=primary')
        dialog.open()

    def _submit_feedback(self, dialog, text):
        dialog.close()
        if self.approval_future and not self.approval_future.done():
            self.approval_future.set_result({"choice": "Feedback", "feedback": text})

    def clear_dynamic_buttons(self):
        self.dynamic_buttons_container.clear()

    def dmp_carousel_prev(self):
        idx = app.storage.general['dmp_carousel_index']
        app.storage.general['dmp_carousel_index'] = max(0, idx - 1)
        self.update_dmp_carousel()

    def dmp_carousel_next(self):
        idx = app.storage.general['dmp_carousel_index']
        count = len(app.storage.general['dmp_carousel_images'])
        app.storage.general['dmp_carousel_index'] = min(count - 1, idx + 1)
        self.update_dmp_carousel()

    def update_dmp_carousel(self):
        imgs = app.storage.general.get('dmp_carousel_images', [])
        idx = app.storage.general.get('dmp_carousel_index', 0)
        if imgs and 0 <= idx < len(imgs):
            self.dmp_plot_image.set_source(imgs[idx])
            self.dmp_counter.set_text(f"{idx + 1} / {len(imgs)}")
            self.dmp_prev_btn.set_enabled(idx > 0)
            self.dmp_next_btn.set_enabled(idx < len(imgs) - 1)
            self.dmp_plot_image.set_visibility(True)
            self.task_video.set_visibility(False)
            if self.simulation_title_label:
                self.simulation_title_label.set_text('DMP TRAJECTORY SIMULATION')
            # Hide overlay when simulation plots are ready
            if self.rollout_overlay:
                self.rollout_overlay.set_visibility(False)
        else:
            self.dmp_plot_image.set_source('')
            self.dmp_counter.set_text("0 / 0")
            self.dmp_prev_btn.set_enabled(False)
            self.dmp_next_btn.set_enabled(False)
            self.dmp_plot_image.set_visibility(False)
            self.task_video.set_visibility(True)
            if self.simulation_title_label:
                self.simulation_title_label.set_text('LAST ROLLOUT')
            # Use timestamp to force video reload
            import time
            self.task_video.set_source(f'/cache/task_video.mp4?v={int(time.time())}')

    async def update_camera_feed(self):
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{API_ENDPOINTS['Camera']}/latest/color")
                if response.status_code == 200:
                    img_data = base64.b64encode(response.content).decode('utf-8')
                    self.camera_image.set_source(f"data:image/png;base64,{img_data}")
        except Exception: pass

    async def home_robot(self):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.get(f"{API_ENDPOINTS['Controller']}/home")
                self._add_log_entry("🏠 Robot homed.")
        except Exception as e: self._add_log_entry(f"⚠️ Homing error: {e}")

    async def return_to_previous(self):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.get(f"{API_ENDPOINTS['Controller']}/return")
                self._add_log_entry("🔙 Returned to previous position.")
        except Exception as e: self._add_log_entry(f"⚠️ Return error: {e}")

# --- Initialize GUI ---
health_monitor = HealthMonitor()
dashboard = RobotDashboard()
dashboard.render()

# --- Background Tasks ---
async def update_ui_state():
    async with ui_state_lock:
        # Logs
        while app.storage.general['log_messages']:
            msg = app.storage.general['log_messages'].pop(0)
            if msg == "[CLEAR]": dashboard.task_log.clear()
            else: dashboard._add_log_entry(msg)
        
        # Progress
        dashboard._update_task_progress_label()
        
        # Images
        if app.storage.general['yolo_image_data']:
            dashboard.yolo_image.set_source(app.storage.general['yolo_image_data'])
            app.storage.general['yolo_image_data'] = None
        
        if app.storage.general.get('dmp_plot_data'):
            dashboard.update_dmp_carousel()
            app.storage.general['dmp_plot_data'] = None
            
        # Confirmation
        if app.storage.general['confirmation_request']:
            req = app.storage.general['confirmation_request']
            app.storage.general['confirmation_request'] = None
            asyncio.create_task(handle_confirmation(req['request_id'], req['options']))

async def update_prompt_log():
    async with ui_state_lock:
        if app.storage.general.get('prompt_dirty'):
            app.storage.general['prompt_dirty'] = False
            dashboard.prompt_log.clear()
            msgs = app.storage.general.get('prompt_messages')
            if msgs:
                with dashboard.prompt_log: ui.html(render_prompt_messages_html(msgs))
            else:
                txt = (app.storage.general.get('prompt_text') or "").rstrip("\n").replace("\n", "<br>")
                with dashboard.prompt_log: ui.html(f'<div class="q-pa-sm text-grey-3 font-mono text-xs">{txt}</div>')
            dashboard.prompt_log_scroll.scroll_to(percent=1.0)

async def handle_confirmation(req_id: str, options: List[str]):
    result = await dashboard.wait_for_option_selection(options)
    if req_id in app._confirmation_futures:
        try: app._confirmation_futures[req_id].set_result(result)
        except Exception: pass
        del app._confirmation_futures[req_id]

ui.timer(HEALTH_CHECK_INTERVAL, health_monitor.update_all_health)
ui.timer(CAMERA_UPDATE_INTERVAL, dashboard.update_camera_feed)
ui.timer(UI_UPDATE_INTERVAL, update_ui_state)
ui.timer(UI_UPDATE_INTERVAL, update_prompt_log)

# --- API Endpoints ---
@app.get("/health")
def health(): return {"status": "healthy"}

@app.post("/api/yolo_result")
async def yolo_result(request: dict):
    img_b64 = request.get("image_base64")
    if not img_b64:
        path = request.get("file_path")
        if path and Path(path).exists():
            with open(path, 'rb') as f: img_b64 = base64.b64encode(f.read()).decode()
    
    if img_b64:
        async with ui_state_lock:
            app.storage.general['yolo_image_data'] = f"data:image/png;base64,{img_b64}"
        return {"status": "success"}
    return {"status": "error", "message": "No image data"}

@app.post("/api/set_subtask")
async def set_subtask(request: dict):
    subtask = request.get("subtask")
    if subtask:
        async with ui_state_lock: app.storage.general['current_task'] = subtask
        return {"status": "success"}
    return {"status": "error"}

@app.post("/api/show_dmp_plots")
async def show_dmp_plots(request: dict):
    img_b64 = request.get("image_base64")
    if img_b64:
        async with ui_state_lock:
            data = f"data:image/png;base64,{img_b64}"
            app.storage.general['dmp_carousel_images'].append(data)
            app.storage.general['dmp_carousel_index'] = len(app.storage.general['dmp_carousel_images']) - 1
            app.storage.general['dmp_plot_data'] = data
        return {"status": "success"}
    return {"status": "error"}

@app.post("/api/add_log")
async def add_log(request: dict):
    msg = request.get("message")
    if msg:
        msg = msg.rstrip("\n").replace("\n", "<br>")
        async with ui_state_lock: app.storage.general['log_messages'].append(msg)
        return {"status": "success"}
    return {"status": "error"}

@app.post("/api/add_prompt")
async def add_prompt(request: dict):
    async with ui_state_lock:
        app.storage.general.update({
            'prompt_messages': request.get("messages"),
            'prompt_text': request.get("text"),
            'prompt_dirty': True
        })
    return {"status": "success"}

@app.post("/api/confirmation")
async def confirmation(request: dict):
    options = request.get("options", [])
    if not options: return {"status": "error"}
    
    req_id = str(uuid.uuid4())
    future = asyncio.Future()
    app._confirmation_futures[req_id] = future
    
    async with ui_state_lock:
        app.storage.general['confirmation_request'] = {'request_id': req_id, 'options': options}
    
    try:
        result = await asyncio.wait_for(future, timeout=300.0)
        return {"status": "success", "result": result}
    except asyncio.TimeoutError:
        if req_id in app._confirmation_futures: del app._confirmation_futures[req_id]
        return {"status": "error", "message": "timeout"}

ui.run(port=8040, title="LMP Dashboard", dark=True)
