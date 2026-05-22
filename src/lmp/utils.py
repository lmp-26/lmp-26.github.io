import numpy as np
from scipy.spatial.transform import Rotation as R

def from_euler_to_quat(euler_angles, format='xyz'):
    y = R.from_euler(format, euler_angles, degrees=False).as_quat()
    return y

def from_quat_to_yaw(quat, format='xyz'):
    if not isinstance(quat, np.ndarray):
        quat = np.array(quat)
    r = R.from_quat(quat)
    angles = r.as_euler(format, degrees=False)
    yaw = angles[2]    
    return yaw