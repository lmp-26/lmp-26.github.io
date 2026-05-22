import time
import pygame
import socket
import numpy as np
from pynput.keyboard import Listener


class Franka(object):

    def __init__(self):
        # self.home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
        self.home = np.array([-0.232867, -0.524729, 0.137301, -2.3478, 0.0455863, 1.85082, 0.64003])

    def connect(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('172.16.0.3', port))
        s.listen()
        conn, addr = s.accept()
        return conn

    def send2gripper(self, conn, command):
        send_msg = "s," + command + ","
        conn.send(send_msg.encode())

    def send2robot(self, conn, qdot, limit=1.0):
        qdot = np.asarray(qdot)
        scale = np.linalg.norm(qdot)
        if scale > limit:
            qdot *= limit/scale
        send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
        if send_msg == '0.,0.,0.,0.,0.,0.,0.':
            send_msg = '0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000'
        send_msg = "s," + send_msg + ","
        conn.send(send_msg.encode())

    def listen2robot(self, conn):
        state_length = 7 + 6 + 42
        # message = str(conn.recv(2048))[2:-2]
        message = str(conn.recv(20480))[2:-2]
        state_str = list(message.split(","))
        for idx in range(len(state_str)):
            if state_str[idx] == "s":
                state_str = state_str[idx+1:idx+1+state_length]
                break
        try:
            state_vector = [float(item) for item in state_str]
        except ValueError:
            return None
        if len(state_vector) is not state_length:
            return None
        state_vector = np.asarray(state_vector)
        states = {}
        states["q"] = state_vector[0:7]
        states["O_F"] = state_vector[7:13]
        states["J"] = state_vector[13:].reshape((7,6)).T
        xyz_lin, R = self.joint2pose(state_vector[0:7])
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        xyz_ang = [alpha, beta, gamma]
        xyz = np.asarray(xyz_lin).tolist() + np.asarray(xyz_ang).tolist()
        states["x"] = np.array(xyz)
        states["angle"] = np.array(xyz_ang)
        return states

    def readState(self, conn):
        while True:
            states = self.listen2robot(conn)
            if states is not None:
                break
        return states

    def xdot2qdot(self, xdot, states):
        J_inv = np.linalg.pinv(states["J"])
        return J_inv @ np.asarray(xdot)

    def joint2pose(self, q):
        def RotX(q):
            return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
        def RotZ(q):
            return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        def TransX(q, x, y, z):
            return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
        def TransZ(q, x, y, z):
            return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        H1 = TransZ(q[0], 0, 0, 0.333)
        H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
        H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
        H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
        H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
        H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
        H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
        # H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
        H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.107 + 0.15)
        T = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
        R = T[:,:3][:3]
        xyz = T[:,3][:3]
        return xyz, R

    def go2position(self, conn, goal=False):
        if type(goal) == bool:
            goal = self.home
        total_time = 20.0
        start_time = time.time()
        states = self.readState(conn)
        dist = np.linalg.norm(states["q"] - goal)
        elapsed_time = time.time() - start_time
        while dist > 0.05 and elapsed_time < total_time:
            qdot = np.clip(goal - states["q"], -0.1, 0.1)
            self.send2robot(conn, qdot)
            states = self.readState(conn)
            dist = np.linalg.norm(states["q"] - goal)
            elapsed_time = time.time() - start_time
        # we need to stop the robot's motion
        states = self.readState(conn)
        qdot = 0.0 * states["q"]
        self.send2robot(conn, qdot)

# used to interrupt the dmp as necessary
# pressing A should pause the execution
# pressing B should resume the execution
# pressing START should terminate the DMP
class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()

    def input(self):
        pygame.event.get()

        A_pressed = self.gamepad.get_button(0)
        B_pressed = self.gamepad.get_button(1)
        START_pressed = self.gamepad.get_button(7)
        return A_pressed, B_pressed, START_pressed
    


class Keyboard:
    def __init__(self):

        listen_keys = ["a", "s", "x", "y", "b"]
        self.key_pressed = {key:False for key in listen_keys}

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):

        try:
            if key.char == "a":
                self.key_pressed["a"] = True
            elif key.char == "s":
                self.key_pressed["s"] = True
            elif key.char == "x":
                self.key_pressed["x"] = True
            elif key.char == "b":
                self.key_pressed["b"] = True
            elif key.char == "y":
                self.key_pressed["y"] = True
        except AttributeError as e:
            pass

    def on_release(self, key):
        try:
            if key.char == "a":
                self.key_pressed["a"] = False
            elif key.char == "s":
                self.key_pressed["s"] = False
            elif key.char == "x":
                self.key_pressed["x"] = False
            elif key.char == "b":
                self.key_pressed["b"] = False
            elif key.char == "y":
                self.key_pressed["y"] = False
        except AttributeError as e:
            pass

    def close(self):
        self.listener.join()
    
    @property
    def a_pressed(self):
        return self.key_pressed["a"]
    
    @property
    def s_pressed(self):
        return self.key_pressed["s"]
    
    @property
    def x_pressed(self):
        return self.key_pressed["x"]
    
    @property
    def b_pressed(self):
        return self.key_pressed["b"]
    
    @property
    def y_pressed(self):
        return self.key_pressed["y"]