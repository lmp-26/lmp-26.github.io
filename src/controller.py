import io
import cv2 
import time
import uvicorn
import platform
import requests
import threading
import numpy as np
from pydantic import BaseModel
import matplotlib.pyplot as plt
from fastapi.responses import Response
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from scipy.spatial.transform import Rotation as R

from utils import Franka
from utils import Keyboard

as_quat = R.as_quat
as_euler = R.as_euler

interface = None
robot = None
conn_robot = None
conn_gripper = None
gripper_open = None
last_start_pose = None

CAMERA_URL = "http://localhost:8010"

_interface_thread = None
_stop_evt = threading.Event()
latest_cmd = None 

def interface_loop():
    global latest_cmd, gripper_open
    while not _stop_evt.is_set():
        if interface.x_pressed:
            print("[-] gripper opening...")
            if not gripper_open:
                robot.send2gripper(conn_gripper, "o")
                gripper_open = True
                time.sleep(2.0)
        if interface.y_pressed:
            print("[-] gripper closing...")
            if gripper_open:
                robot.send2gripper(conn_gripper, "c")
                gripper_open = False
                time.sleep(2.0)

        time.sleep(0.01) 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and processor once at startup, cleanup on shutdown"""
    global interface, robot, conn_robot, conn_gripper, gripper_open, _interface_thread

    print(f"Initializing Robot...")
    if platform.system() != "Darwin":
        interface = Keyboard()
        robot = Franka()
        print('[*] Connecting to robot...')
        conn_robot = robot.connect(8080)
        print('[*] Connecting to gripper...')
        conn_gripper = robot.connect(8081)
        # home robot and gripper
        # robot.send2gripper(conn_gripper, "o")
        # robot.go2position(conn_robot)
        gripper_open = True
    else:
        print("Running on macOS - skipping robot connection for debugging.")

    _stop_evt.clear()
    _interface_thread = threading.Thread(target=interface_loop, daemon=True)
    _interface_thread.start()
    print('[*] Connection complete')
    
    yield

    print("Shutting down...")
    _stop_evt.set()
    if _interface_thread is not None:
        _interface_thread.join(timeout=1.0)

# Initialize FastAPI app with lifespan
app = FastAPI(title="DMP Robot API", lifespan=lifespan)

class RunDMPRequest(BaseModel):
    weights: list
    goal: list
    beta_z: float | None = None
    alpha_z: float | None = None
    tau: float | None = None
    forcing_gain: float | None = None
    alpha_x: float | None = None
    dynamic: bool = False  
    llm_monitoring: bool = False
    height: float = 0.0

class DMPResponse(BaseModel):
    status: str
    success: bool
    reason: str

# needs the weights (7 x 11 matrix) and goal (7 x 1 vector)
class DMP(object):

    # hyperparameters
    # tau is the total "time" to complete the motion
    # beta_z shapes how quickly the dmp converges to goal
    def __init__(self, basis_type='gaussian', dmp_type='position'):
        self.basis_type = basis_type  # 'gaussian' or 'step'
        self.dmp_type = dmp_type  # 'position' or 'gripper'
        self.beta_z = 5.2  # spring stiffness
        self.alpha_z = 4 * self.beta_z  # damping coefficient (alpha = 4*beta for critical damping)
        self.tau = 1.0
        self.forcing_gain = 2.0  # amplify forcing function contribution
        self.alpha_x = 4.5  # Exponential decay rate for canonical system (higher = faster decay at end)
        if self.dmp_type == 'gripper':
            self.beta_z = 10.5
            self.alpha_z = 0.8  * self.beta_z
            self.forcing_gain = 8.0 
            self.alpha_x = 0.0
        self.z = 0.0
        self.x = 1.0
        self.t = 0.0  # time progression

    # set the step size
    # this governs how quickly we move through the dmp
    def set_step(self, step_size):
        self.step_size = step_size

    # set the start position
    def set_start(self, start):
        self.start = start
        self.xi = start

    # set the goal position
    def set_goal(self, goal):
        self.goal = goal

    # check the progress
    def get_progress(self):
        return 1.0 - self.x

    # set the weight values
    # expected: a vector of length N (11)
    # weight[0] corresponds to start of the trajectory,
    # weight[10] corresponds to the end of the trajectory
    def set_weights(self, weights):
        self.weights = weights

    # gaussian basis function
    def gaussian_basis(self, x, h, c):
        return np.exp(-h * (x - c)**2)

    # step basis function
    def step_basis(self, x, lower, upper):
        return 1.0 if lower <= x < upper else 0.0

    # forcing function
    # basis functions are evaluated based on normalized time (trajectory progress)
    def forcing_function(self):
        N = 11
        # Use normalized time as phase variable (0 to 1)
        phase = min(self.t / self.tau, 1.0)
        # Multiply by canonical system x to ensure forcing decays to 0 at end
        x_decay = self.x

        if self.basis_type == 'gaussian':
            # h = 150.0
            c = np.linspace(0, 1, N)
            dc = c[1] - c[0]  
            sigma = 0.25 * dc  # std
            h = 1.0 / (sigma**2)  # Convert sigma
            basis = np.array([self.gaussian_basis(phase, h, c[idx]) for idx in range(N)])
            return np.sum(self.weights * basis) / np.sum(basis) * x_decay

        elif self.basis_type == 'step':
            boundaries = np.linspace(0, 1, N + 1)
            basis = np.array([self.step_basis(phase, boundaries[idx], boundaries[idx+1])
                            for idx in range(N)])
            # Find active segment (only one should be active)
            active_idx = np.argmax(basis)
            if np.any(basis):
                return self.weights[active_idx] * x_decay
            else:
                return 0.0

        else:
            raise ValueError(f"Unknown basis_type: {self.basis_type}")

    # step to get the next position along the trajectory
    def step(self):
        # Increment time first
        self.t += self.step_size

        # Update canonical system with reversed exponential decay
        # We want x to stay near 1.0 for longer (high forcing influence)
        # then rapidly decay to 0 near the end (goal convergence guarantee)
        # Using reversed exponential: starts slow, ends fast
        normalized_time = min(self.t / self.tau, 1.0)  # Clamp to [0, 1]
        # This formula keeps x near 1 early, then drops exponentially to 0
        self.x = np.exp(-self.alpha_x * normalized_time**3)

        # Update transformation system
        # Standard DMP: τ·ż = α_z·β_z·(g - y) - α_z·z + f
        # For critical damping: α_z = 4·β_z
        zdot = (self.alpha_z * self.beta_z * (self.goal - self.xi) - self.alpha_z * self.z + self.forcing_gain * self.forcing_function()) / self.tau
        xidot = self.z / self.tau
        self.z += zdot * self.step_size
        self.xi += xidot * self.step_size

        return self.xi
    
    def settle_step(self, k_settle=5.0):
        self.x -= self.step_size
        f = 0.0

        alpha = self.alpha_z * k_settle
        beta  = self.beta_z  * k_settle

        zdot = (alpha * (beta * (self.goal - self.xi) - self.z) + f) / self.tau
        xidot = self.z / self.tau

        self.z  += zdot  * self.step_size
        self.xi += xidot * self.step_size
        return self.xi


# function to execute DMP
def execute_dmp(dmps, robot, conn_robot, conn_gripper, interface, gripper_open, k_p=1.2, 
                track_goal_position = False, start_angle = None, height = 0.0, llm_monitoring = False):
    pause_dmp = False
    steps = 0
    new_goal_state = []
    for dmp in dmps:
        new_goal_state.append(dmp.goal)
    new_goal_state = np.array(new_goal_state)
    while dmps[0].x > 0.0112:
        if track_goal_position:
            # need to update the goal position of the dmp
            assert start_angle is not None, "start_angle need to be provided if tracking object position is required."
            try:
                response = requests.get(f"{CAMERA_URL}/track_object", timeout=0.05)
                if response.status_code == 200:
                    data = response.json()
                    tracked_pos = data.get("position")
                    if tracked_pos is not None:
                        new_goal_state = np.array(tracked_pos)
                        new_goal_state[3] = new_goal_state[3] + start_angle
                        new_goal_state[2] = new_goal_state[2] + height
                        for idx, dmp in enumerate(dmps):
                            dmp.set_goal(new_goal_state[idx])
            except Exception:
                pass

        if llm_monitoring:
            try:
                # Poll for updated weights from LLM generator (port 8031)
                llm_response = requests.get("http://localhost:8031/get_current_weights", timeout=0.05)
                if llm_response.status_code == 200:
                    llm_data = llm_response.json()
                    if llm_data.get("weights"):
                        new_weights = np.array(llm_data["weights"])
                        # Apply same scaling factors as in orchestrator
                        new_weights[:-1] = new_weights[:-1] * 20.0
                        new_weights[-2] = new_weights[-2] * 10.0
                        new_weights[-1] = new_weights[-1] * 10.0
                        for idx, dmp in enumerate(dmps):
                            dmp.set_weights(new_weights[idx])
            except Exception:
                pass

        # get robot state and any joystick inputs
        state = robot.readState(conn_robot)
        if interface.a_pressed:
            print("[-] Pausing...")
            pause_dmp = True
        if interface.b_pressed:
            print("[-] Resuming...")
            pause_dmp = False
        if interface.s_pressed:
            print("[-] Terminated")
            return False, gripper_open
        # do not update dmp if paused
        if pause_dmp:
            continue
        steps += 1
        # get desired position
        xi = [dmp.step() for dmp in dmps]
        xi = np.array(xi)
        # check if a command to open / close the gripper is sent
        if xi[-1] > 0.5 and gripper_open:
            # close the gripper
            gripper_open = False
            print("[*] Closing gripper...")
            robot.send2gripper(conn_gripper, "c")
            time.sleep(2.0)
        elif xi[-1] < 0.5 and not gripper_open:
            # open the gripper
            gripper_open = True
            print("[*] Opening gripper...")
            robot.send2gripper(conn_gripper, "o")
            time.sleep(2.0)
        # move the robot arm
        xdot = np.zeros(6)
        # print(f"gripper state: {xi[-1]}")
        print(f"Current end effector position: {state['x'][0:3]}")
        print(f"Desired end effector position: {xi}")
        xdot[0:3] = k_p * (xi[0:3] - state["x"][0:3])
        xdot[-1] = k_p * (xi[3] - state["angle"][-1])
        # set operating boundaries, x should be smaller than 0.7, z should be larger than 0.02
        if state["x"][0] > 0.8 and xdot[0] > 0.0:
            xdot[0] = 0.0
        if state["x"][2] < 0.065 and xdot[2] < 0.0:
            xdot[2] = 0.0
        if state["angle"][-1] > 3.12:
            xdot[-1] = 0.0
        qdot = robot.xdot2qdot(xdot, state)
        robot.send2robot(conn_robot, qdot)
    return True, gripper_open


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/run_dmp", response_model=DMPResponse)
async def run_dmp(request: RunDMPRequest):
    global interface, robot, conn_robot, conn_gripper, gripper_open, last_start_pose
    
    last_start_pose = robot.readState(conn_robot)["q"].copy()

    # Use gaussian for first 3 DOFs, step function for gripper (last DOF)
    dmps = [DMP(basis_type='gaussian') for _ in range(3)] # + [DMP(basis_type='step')]
    dmps.append(DMP()) # rotation
    dmps.append(DMP(basis_type='step', dmp_type='gripper'))  # last DOF is gripper with step basis
    state = robot.readState(conn_robot)
    start_xyz = state["x"][:3]
    start_angle = state["angle"][-1]
    start_state = np.array(start_xyz.tolist() + [start_angle] + [1 - (gripper_open == True)]) 

    # import values for dmps here
    goal_state = np.array(request.goal)
    goal_state[-2] = goal_state[-2] + start_angle  # Adjust goal angle to be relative to current angle
    weights = np.array(request.weights) 

    # set the dmp values
    for idx, dmp in enumerate(dmps):
        dmp.set_start(start_state[idx])
        dmp.set_goal(goal_state[idx])
        dmp.set_weights(weights[idx, :])
        dmp.set_step(0.003)

    # run the dmp
    print(f"gripper is currently set to {gripper_open}")
    outcome, gripper_open = execute_dmp(
        dmps, robot, conn_robot, conn_gripper, interface, gripper_open, 
        track_goal_position=request.dynamic, 
        start_angle=start_angle, 
        height=request.height,
        llm_monitoring=request.llm_monitoring
    )
    if outcome:
        # we have completed the dmp without interruptions
        return DMPResponse(status="DMP completed successfully", success=True, reason="none")
    else:
        # we had to cancel during the dmp
        return DMPResponse(status="DMP failed for some reason", success=False, reason="interrupted")

@app.post("/simulate_dmp")
async def simulate_dmp(request: RunDMPRequest):
    global robot, conn_robot, gripper_open

    weights = np.array(request.weights)
    goal_state = np.array(request.goal)
    num_dofs = weights.shape[0]

    # gaussian for all except last DOF which uses step function
    dmps = [DMP(basis_type='gaussian') for _ in range(num_dofs - 2)] # + [DMP(basis_type='step')]
    dmps.append(DMP())
    dmps.append(DMP(basis_type='step', dmp_type='gripper'))  # last DOF is gripper with step basis

    for dmp in dmps:
        if request.beta_z is not None:
            dmp.beta_z = request.beta_z
        if request.alpha_z is not None:
            dmp.alpha_z = request.alpha_z
        if request.tau is not None:
            dmp.tau = request.tau
        if request.forcing_gain is not None:
            dmp.forcing_gain = request.forcing_gain
        if request.alpha_x is not None:
            dmp.alpha_x = request.alpha_x

    # Get start state from robot or use dummy state
    if conn_robot is not None:
        # Get actual robot position
        state = robot.readState(conn_robot)
        start_xyz = state["x"][0:3]
        start_angle = state["angle"][-1]
        print(f"angle: {start_angle}")
        start_state = np.array(start_xyz.tolist() + [start_angle] + [1 - (gripper_open == True)])  # Gripper state: 0 for open, 1 for closed
        goal_state[-2] = goal_state[-2] + start_angle  # Adjust goal angle to be relative to current angle
    else:
        # Use dummy state when robot is not connected
        dummy_xyz = [0.2, 0.0, 0.5]
        start_state = np.array(dummy_xyz + [0.0])

    if len(start_state) != num_dofs:
        start_state = start_state[:num_dofs]
    for idx, dmp in enumerate(dmps):
        dmp.set_start(start_state[idx])
        dmp.set_goal(goal_state[idx])
        dmp.set_weights(weights[idx, :])
        dmp.set_step(0.01)  # Step size for simulation

    # Simulate DMP trajectory with fixed time steps
    time_points = []  # Track actual time
    canonical_system = []  # x values (1 to 0)
    trajectories = [[] for _ in range(num_dofs)]  # One trajectory per DOF

    # Run for fixed duration based on tau
    max_time = 1.0  # Run for tau duration (1.0 by default)
    step_size = 0.01
    num_steps = int(max_time / step_size)

    for step in range(num_steps):
        time_points.append(dmps[0].t)
        for idx, dmp in enumerate(dmps):
            trajectories[idx].append(dmp.xi)
        canonical_system.append(dmps[0].x)

        for dmp in dmps:
            dmp.step()

    print(f"Simulation completed: {len(time_points)} time steps")
    print(f"Final time: {time_points[-1]:.3f}, Final canonical x: {canonical_system[-1]:.6f}")

    # basis functions for visualization
    N_basis = weights.shape[1]  # Number of basis functions
    N_samples = 100  # Number of samples along time axis
    time_samples = np.linspace(0.0, max_time, N_samples)

    # trajectories on left, basis functions on right
    fig, axes = plt.subplots(num_dofs, 2, figsize=(16, 2*num_dofs))

    # if axes not a 2D array
    if num_dofs == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_dofs):
        # Left column: Plot trajectory vs TIME
        if idx == num_dofs - 1:
            axes[idx, 0].plot(time_points, trajectories[idx], linewidth=2, marker='o', markersize=3, label='Step function')
        else:
            axes[idx, 0].plot(time_points, trajectories[idx], linewidth=2, label='Gaussian')

        axes[idx, 0].set_ylabel(f'DOF {idx}')
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].set_xlim(0.0, time_points[-1])  # X-axis: time from 0 to max

        # Only add x-label to bottom subplot
        if idx == num_dofs - 1:
            axes[idx, 0].set_xlabel('Time (s)')

        # Right column: Plot weighted basis functions vs TIME
        # Calculate basis functions directly to show how input weights map to time
        h = 150.0
        centers = np.linspace(0, 1, N_basis)  # centers from 0 (start) to 1 (end)

        for input_idx in range(N_basis):
            # Input weight at index input_idx should affect motion at phase = input_idx/(N_basis-1)
            # This weight is stored flipped: self.weights[N_basis - 1 - input_idx]
            # weight_value = dmps[idx].weights[N_basis - 1 - input_idx]
            weight_value = dmps[idx].weights[input_idx] 

            # Calculate this weight's basis contribution over time
            weighted_basis = []
            for t_sample in time_samples:
                phase = min(t_sample / dmps[idx].tau, 1.0)
                x_decay = np.exp(-dmps[idx].alpha_x * phase**3)

                # basis centered at the input index's corresponding phase
                center = centers[input_idx]
                basis_val = np.exp(-h * (phase - center)**2)

                # Total basis normalization (sum of all basis functions at this phase)
                all_basis = np.array([np.exp(-h * (phase - c)**2) for c in centers])
                normalization = np.sum(all_basis)

                # Weighted contribution
                weighted_basis.append(weight_value * basis_val / normalization * x_decay)

            axes[idx, 1].plot(time_samples, weighted_basis, linewidth=1.5, alpha=0.7, label=f'w{input_idx}')

        # plot the total forcing function(sum of all weighted basis functions)
        total_forcing = []
        for t_sample in time_samples:
            phase = min(t_sample / dmps[idx].tau, 1.0)
            x_decay = np.exp(-dmps[idx].alpha_x * phase**3)

            # Calculate total forcing as sum of all weighted basis contributions
            total = 0.0
            all_basis = np.array([np.exp(-h * (phase - c)**2) for c in centers])
            normalization = np.sum(all_basis)

            for input_idx in range(N_basis):
                # weight_value = dmps[idx].weights[N_basis - 1 - input_idx]
                weight_value = dmps[idx].weights[input_idx]
                center = centers[input_idx]
                basis_val = np.exp(-h * (phase - center)**2)
                total += weight_value * basis_val / normalization * x_decay

            total_forcing.append(total)

        axes[idx, 1].plot(time_samples, total_forcing, linewidth=2.5, color='black', alpha=0.9, label='Total', linestyle='--')

        axes[idx, 1].set_ylabel(f'Weighted Basis (DOF {idx})')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_xlim(0.0, max_time)  # X-axis from 0 to max_time

        # only add x-label to bottom subplot
        if idx == num_dofs - 1:
            axes[idx, 1].set_xlabel('Time (s)')

    plt.tight_layout()

    # Convert figure to image buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_array = np.frombuffer(buf.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    buf.close()
    plt.close(fig)

    success, buffer = cv2.imencode('.png', img_bgr)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(
        content=buffer.tobytes(),
        media_type="image/png"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy"
    }

@app.get("/return")
async def return_previous():
    # home robot and gripper
    global conn_gripper, conn_robot, robot, last_start_pose, gripper_open
    robot.send2gripper(conn_gripper, "o")
    gripper_open = True
    if last_start_pose is not None:
        robot.go2position(conn_robot, goal=last_start_pose)
    else:
        robot.go2position(conn_robot)
    return {
        "status": "ok"
    }

@app.get("/home")
async def go_home():
    # home robot and gripper
    global conn_gripper, conn_robot, robot, gripper_open
    gripper_open = True
    robot.send2gripper(conn_gripper, "o")
    robot.go2position(conn_robot)
    return {
        "status": "ok"
    }

@app.get("/get_state")
async def get_state():
    global robot, conn_robot
    if conn_robot is not None:
        state = robot.readState(conn_robot)
        state = np.array(state["x"][:3].tolist() + [state["angle"][-1]] + [1 - (gripper_open == True)])  # Combine position, angle, and gripper state
        return JSONResponse(content={"state": state.tolist()})
    else:
        raise HTTPException(status_code=503, detail="Robot not connected")
    
@app.get("/get_joint_state")
async def get_joint_state():
    global robot, conn_robot, gripper_open
    if conn_robot is not None:
        state = robot.readState(conn_robot)
        state = np.array(state["q"].tolist() + [gripper_open == True])  # Combine joint position, and gripper state
        return JSONResponse(content={"state": state.tolist()})
    else:
        raise HTTPException(status_code=503, detail="Robot not connected")

@app.post("/open_gripper")
async def open_gripper():
    global robot, conn_gripper, gripper_open
    if not gripper_open:
        robot.send2gripper(conn_gripper, "o")
        gripper_open = True
        time.sleep(2.0)
    return {
        "status": "ok",
    }

@app.post("/close_gripper")
async def close_gripper():
    global robot, conn_gripper, gripper_open
    if gripper_open:
        robot.send2gripper(conn_gripper, "c")
        gripper_open = False
        time.sleep(2.0)
    return {
        "status": "ok",
    }

@app.post("/open_gripper_wo_delay")
async def open_gripper_wo_delay():
    global robot, conn_gripper, gripper_open
    if not gripper_open:
        robot.send2gripper(conn_gripper, "o")
        gripper_open = True
    return {
        "status": "ok",
    }

@app.post("/close_gripper_wo_delay")
async def close_gripper_wo_delay():
    global robot, conn_gripper, gripper_open
    if gripper_open:
        robot.send2gripper(conn_gripper, "c")
        gripper_open = False
    return {
        "status": "ok",
    }

from pydantic import BaseModel
from typing import List

class ActionRequest(BaseModel):
    action: list

class TrajectoryRequest(BaseModel):
    trajectory: List[List[float]]

@app.post("/execute_action")
async def execute_action(req: ActionRequest):
    global robot, conn_robot, gripper_open, interface
    qdot = np.array(req.action)
    robot.send2robot(conn_robot, qdot)
    return {
        "status": "ok",
    }

@app.post("/execute_trajectory")
async def execute_trajectory(req: TrajectoryRequest):
    global robot, conn_robot, gripper_open, interface

    if conn_robot is None:
        raise HTTPException(status_code=503, detail="Robot not connected")
    trajectory = req.trajectory
        
    for i, point in enumerate(trajectory):
        # point should be a list of 4 values: [x, y, z, angle]
        if len(point) != 4:
            raise HTTPException(status_code=400, detail="Each trajectory point must have 4 values: [x, y, z, angle]")
        
        x_desired = np.array(point[0:4])  # Desired position and angle

        # Get current state
        state = robot.readState(conn_robot)
        x_current = np.array(state["x"][:3].tolist() + [state["angle"][-1]])  # Current position and angle

        # move towards desired position iteratively
        print(f"Moving to point: {x_desired}")
        print(f"Current position: {x_current}")
        steps = 0
        while np.linalg.norm(x_desired - x_current) > 0.015 and steps < 10:
            if interface.s_pressed:
                raise HTTPException(status_code=400, detail="Trajectory execution interrupted by user")
            print(f"error: {np.linalg.norm(x_desired - x_current)}")
            xdot = np.zeros(6)
            scale = 1.0 if i < 5 else 2.0  
            xdot[0:3] = scale * (x_desired[:3] - state["x"][:3])
            xdot[-1] = scale * (x_desired[-1] - state["angle"][-1])
            if state["x"][0] > 0.8 and xdot[0] > 0.0:
                xdot[0] = 0.0
            if state["x"][2] < 0.065 and xdot[2] < 0.0:
                xdot[2] = 0.0
            if state["angle"][-1] > 3.12:
                xdot[-1] = 0.0

            qdot = robot.xdot2qdot(xdot, state)
            robot.send2robot(conn_robot, qdot)

            # time.sleep(0.01)  # Small delay for control loop
            steps += 1

            # Update current state
            state = robot.readState(conn_robot)
            x_current = np.array(state["x"][:3].tolist() + [state["angle"][-1]])

    return {
        "status": "ok",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8030)
