import os
import re
import cv2
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image

_global_figures = {}

def show_persistent_plot(title: str, image_rgb: np.ndarray):
    """
    Displays an image in a persistent window using Matplotlib.
    Matches the user's preferred 'simple' implementation.
    """
    try:
        global _global_figures
        fig = _global_figures.get(title)
        
        # Check if figure exists and is still open
        fig_number = getattr(fig, "number", None)
        if fig is None or fig_number is None or not plt.fignum_exists(fig_number):
            fig = plt.figure(num=title)
            _global_figures[title] = fig

        fig.clf()
        ax = fig.add_subplot(111)
        ax.imshow(image_rgb)
        ax.set_title(title)
        ax.axis("off")
        fig.canvas.draw_idle()
        fig.show()
        plt.pause(0.001)
    except Exception:
        # Catch threading/GUI errors (e.g. "main thread is not in main loop")
        # without crashing the server or tracking loop
        pass

def load_c2r_matrix() -> np.ndarray | None:
    try:
        matrix_path = os.path.join(os.path.dirname(__file__), "C2R.npy")
        C2R = np.load(matrix_path)
    except Exception:
        try:
            C2R = np.load("C2R.npy")
        except Exception:
            return None
    C2R = np.asarray(C2R, dtype=np.float64)
    if C2R.shape != (4, 4):
        return None
    return C2R

FILTER_PHRASES = [
    "robot", "arm", "surface", "frame", "gripper", "table",
    "tabletop", "background", "container", "box", "cable", "support",
    "curtain", "cloth", "cable", "power", "cord", "wall", "floor",
    "camera", "base", "light", "warning", "pole", "desk", 
    "button", "joint", "fabric", "control", "end"
]

def parse_response(response_text: str):
    json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        detections = json.loads(json_str)

        filtered_detections = []
        for detection in detections:
            label = detection.get('label', '')
            skip = False
            for word in label.lower().split():
                for filter_phrase in FILTER_PHRASES:
                    if filter_phrase in word:
                        skip = True
                        break

            if not skip:
                point = detection.get('point', [0, 0])
                fx = 3840.0 / 1000.0 * point[1]
                fy = 2160.0 / 1000.0 * point[0]
                detection['point'] = [fx, fy]
                filtered_detections.append(detection)

        return filtered_detections
    return None

def uvz_to_xyz(uvz, fx, fy, cx, cy):
    uvz = np.asarray(uvz, dtype=np.float64)
    u = uvz[:, 0]
    v = uvz[:, 1]
    z = uvz[:, 2]
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    return np.stack([X, Y, Z], axis=1)

def xyz_to_duv(xyz: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    xyz = np.asarray(xyz, dtype=np.float64)
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]
    Z = np.maximum(Z, 1e-6)
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return u, v

def draw_object_pc_on_color(img, pts_duvz, stride=3, radius=1, alpha=0.45):
    if pts_duvz is None or len(pts_duvz) == 0:
        return
    Hc, Wc = img.shape[:2]
    pts = pts_duvz[::max(1, int(stride))]
    overlay = img.copy()
    col = (0, 255, 0) 
    for u_d, v_d, _z in pts:
        px, py = float(u_d), float(v_d)
        ix = int(np.clip(round(px), 0, Wc - 1))
        iy = int(np.clip(round(py), 0, Hc - 1))
        cv2.circle(overlay, (ix, iy), int(radius), col, -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0, img)

def draw_wireframe(img, uv, label):
    maroon = (0, 0, 128)
    edges = [(0, 3), (3, 5), (5, 2), (2, 0), (1, 6), (6, 4), (4, 7), (7, 1), (4, 5), (6, 3), (7, 2), (1, 0)]
    for a, b in edges:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), maroon, 2, lineType=cv2.LINE_AA)
    top = int(np.argmin(uv[:, 1]))
    tx, ty = uv[top]
    cv2.putText(img, label, (tx, max(0, ty - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, maroon, 3, cv2.LINE_AA)

def draw_uv_indices(img, uv):
    for i, (x, y) in enumerate(uv):
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, str(i), (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, str(i), (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
        
def mask_filter(mask_u8, outline_px=2):
    m = (mask_u8 > 0).astype(np.uint8)
    if outline_px <= 0: return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(m, k, iterations=int(outline_px))

def pc_filter_xyz_fast(pts_xyz, mad_k=2.5, voxel=5.0, radius=12.0, min_nb=10, nb_neighbors=40, std_ratio=1.0):
    pts = np.asarray(pts_xyz, dtype=np.float64)
    z = pts[:, 2]
    med = np.median(z)
    mad = np.median(np.abs(z - med)) + 1e-6
    pts = pts[np.abs(z - med) <= (mad_k * 1.4826 * mad)]
    if pts.shape[0] < 200: return pts.astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if voxel > 0: pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
    pcd, _ = pcd.remove_radius_outlier(nb_points=int(min_nb), radius=float(radius))
    if len(pcd.points) < 200: return np.asarray(pcd.points, dtype=np.float32)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
    return np.asarray(pcd.points, dtype=np.float32)

def get_bbox_from_mask(mask, depth_mm, annotated, kernel, fx, fy, cx, cy, C2R_np):
    Hc, Wc = annotated.shape[:2]
    Hd, Wd = depth_mm.shape[:2]
    
    num_init = np.sum(mask > 0)
    try:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = mask_filter(mask, outline_px=2)
    except:
        return None

    ys, xs = np.where(mask.astype(bool))
    num_morph = xs.size
    if num_morph < 10: 
        print(f"[BBOX DEBUG] Mask too small after morphology: {num_morph} (init: {num_init})")
        return None
    
    ud_i = np.clip(np.round(xs).astype(np.int32), 0, Wd - 1)
    vd_i = np.clip(np.round(ys).astype(np.int32), 0, Hd - 1)
    z = depth_mm[vd_i, ud_i].astype(np.float32)
    
    valid = (z >= 200) & (z <= 20000)
    num_valid = np.sum(valid)
    if not np.any(valid): 
        print(f"[BBOX DEBUG] No valid depth points in mask range. Total points: {num_morph}")
        return None
        
    pts_duvz = np.stack([ud_i[valid].astype(np.float32), vd_i[valid].astype(np.float32), z[valid]], axis=1)
    if pts_duvz.shape[0] < 10: 
        print(f"[BBOX DEBUG] Too few valid depth points after filtering: {pts_duvz.shape[0]}")
        return None

    draw_object_pc_on_color(annotated, pts_duvz, stride=4, radius=1, alpha=0.35)
    pts_xyz = uvz_to_xyz(pts_duvz, fx, fy, cx, cy)
    pts_xyz = pc_filter_xyz_fast(pts_xyz, mad_k=2.5)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_xyz.astype(np.float64))
    obb = pcd.get_minimal_oriented_bounding_box()
    corners_xyz = np.asarray(obb.get_box_points(), dtype=np.float64)
    center_xyz = np.asarray(pcd.get_center(), dtype=np.float64)
    u_c, v_c = xyz_to_duv(center_xyz[None, :], fx, fy, cx, cy)
    px = float(np.clip(int(round(u_c[0])), 0, Wc - 1))
    py = float(np.clip(int(round(v_c[0])), 0, Hc - 1))
    center_xyz_m = center_xyz / 1000.0
    center_robot_np_h = C2R_np @ np.array([center_xyz_m[0], center_xyz_m[1], center_xyz_m[2], 1.0], dtype=np.float64)
    center_robot_np = (center_robot_np_h[:3] / center_robot_np_h[3]).tolist()
    u_c, v_c = xyz_to_duv(corners_xyz, fx, fy, cx, cy)
    corners_pxpyz = []
    corners_uv = []
    for (u_d, v_d, z_corner) in zip(u_c, v_c, corners_xyz[:, 2]):
        px_c = float(np.clip(int(round(u_d)), 0, Wc - 1))
        py_c = float(np.clip(int(round(v_d)), 0, Hc - 1))
        corners_pxpyz.append([px_c, py_c, float(z_corner)])
        corners_uv.append([int(px_c), int(py_c)])
    return corners_xyz, np.asarray(corners_uv, dtype=np.int32), corners_pxpyz, center_robot_np

def generate_bbox(depth_mm, color_bgr, detections, langsam_model, fx, fy, cx, cy, query: str = ""):
    C2R_np = load_c2r_matrix()
    assert C2R_np is not None, "C2R transformation matrix not found. Please ensure C2R.npy is in the working directory."
    
    Hc, Wc = color_bgr.shape[:2]
    Hd, Wd = depth_mm.shape[:2]
    annotated = color_bgr.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image_pil = Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
    results = []
    label_cache = {}
    for det in detections:
        label = det.get("label", "object")
        x, y = det.get("point", [0, 0])
        x, y = float(x), float(y)
        if label not in label_cache:
            try:
                if query == "":
                    pred = langsam_model.predict([image_pil], [label])
                else:
                    pred = langsam_model.predict([image_pil], [label], box_threshold=0.15)
            except: continue
            res = pred[0]
            masks_l = res["masks"]
            if query != "":
                label_cache[label] = masks_l
            else:
                xi, yi = int(np.clip(round(x), 0, Wc - 1)), int(np.clip(round(y), 0, Hc - 1))
                chosen = None
                for m in masks_l:
                    if (np.asarray(m) > 0).astype(np.uint8)[yi, xi] > 0:
                        chosen = (np.asarray(m) > 0).astype(np.uint8)
                        break
                if chosen is None: 
                    # chosen = (np.asarray(masks_l[0]) > 0).astype(np.uint8)
                    chosen = 0
                label_cache[label] = [chosen]
        else: continue
        
        masks_to_process = label_cache[label]
        if len(masks_to_process) == 0: continue
        for idx, m in enumerate(masks_to_process):
            mask = (np.asarray(m) > 0).astype(np.uint8)
            res = get_bbox_from_mask(mask, depth_mm, annotated, kernel, fx, fy, cx, cy, C2R_np)
            if res is None: continue
            corners_xyz, corners_uv, corners_pxpyz, center_robot_np = res
            cv2.circle(annotated, (int(round(x)), int(round(y))), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            draw_wireframe(annotated, corners_uv, label)
            draw_uv_indices(annotated, corners_uv)
            results.append({
                "label": f"{label}_{idx}" if len(masks_to_process) > 1 else label,
                "corners": corners_xyz,
                "pixel_corners": corners_pxpyz,
                "centers": center_robot_np,
                "mask": mask
            })
    return results, annotated

def clean_bbox_dict(obbs):
    C2R_np = load_c2r_matrix()
    assert C2R_np is not None, "C2R transformation matrix not found. Please ensure C2R.npy is in the working directory."
    corner_dict = {}
    pixel_corner_dict = {}
    for item in obbs:
        label = item.get("label", "object")
        corner_dict[label] = item.get("corners", None)
        pixel_corner_dict[label] = item.get("pixel_corners", None)
    bbox = {}

    for obj, corners in corner_dict.items():
        corners = np.asarray(corners, dtype=np.float64) / 1000.0
        pixel_corners = np.asarray(pixel_corner_dict[obj], dtype=np.float64)

        if corners.shape != (8, 3):
            continue

        homo_np = np.hstack([corners, np.ones((8, 1), dtype=np.float64)])
        corners_np_h = (C2R_np @ homo_np.T).T
        corners_robot_np = corners_np_h[:, :3] / corners_np_h[:, 3:4]

        p0 = corners_robot_np[0]
        e01 = corners_robot_np[1] - p0
        e02 = corners_robot_np[2] - p0
        e03 = corners_robot_np[3] - p0

        cp0 = corners[0]
        ce01 = corners[1] - cp0
        ce02 = corners[2] - cp0
        ce03 = corners[3] - cp0

        E = np.stack([e01, e02, e03], axis=0)
        CE = np.stack([ce01, ce02, ce03], axis=0)
        L = np.linalg.norm(E, axis=1) + 1e-9
        CL = np.linalg.norm(CE, axis=1) + 1e-9

        horiz_score = np.abs(E[:, 2]) / L
        idx = np.argsort(horiz_score)
        i_a, i_b = int(idx[0]), int(idx[1])

        va, vb = E[i_a], E[i_b]
        cla, clb = float(CL[i_a]), float(CL[i_b])

        if cla >= clb:
            v_len, length, width = va, cla, clb
        else:
            v_len, length, width = vb, clb, cla

        i_h = int(idx[2])
        height = float(CL[i_h])

        vx, vy = float(v_len[0]), float(v_len[1])
        angle_ccw = float(np.arctan2(vy, vx))

        if angle_ccw > np.pi / 2:
            angle_ccw -= float(np.pi)
        if angle_ccw <= -np.pi / 2:
            angle_ccw += float(np.pi)

        bbox[obj] = {
            "height": height,
            "width": float(width),
            "length": float(length),
            "angle": float(angle_ccw),
            "corners": corners_robot_np
        }

    return bbox

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj

class VisionDetector:
    def __init__(self, fx, fy, cx, cy, distortion, marker_size, inpainting_factor=0.1):  
        self.camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.distortion = np.array(distortion)
        self.marker_size = marker_size
        self.inpainting_factor = inpainting_factor
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    def estimatePoseSingleMarkers(self, corners):
        marker_points = np.array([[-self.marker_size / 2, self.marker_size / 2, 0],
                                [self.marker_size / 2, self.marker_size / 2, 0],
                                [self.marker_size / 2, -self.marker_size / 2, 0],
                                [-self.marker_size / 2, -self.marker_size / 2, 0]], dtype=np.float32)
        trash = []; rvecs = []; tvecs = []
        for c in corners:
            nada, R, t = cv2.solvePnP(marker_points, c, self.camera_intrinsics, self.distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(R); tvecs.append(t); trash.append(nada)
        return rvecs, tvecs, trash

    def pose_vectors_to_cart(self, rvecs, tvecs, ids=None):
        try: C2R = load_c2r_matrix()
        except: C2R = None
        camera_poses = []; robot_poses = []
        if rvecs is None or tvecs is None: return camera_poses, robot_poses
        marker_ids = np.asarray(ids).reshape(-1).tolist() if ids is not None else None
        for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            r = np.asarray(rvec, dtype=np.float64).reshape(-1, 3)[0]
            t = np.asarray(tvec, dtype=np.float64).reshape(-1, 3)[0]
            R, _ = cv2.Rodrigues(r.reshape(3, 1))
            cam_pose = {"x": float(t[0]), "y": float(t[1]), "z": float(t[2]), "rvec": r.tolist(), "R": R.tolist()}
            if marker_ids and idx < len(marker_ids): cam_pose["id"] = int(marker_ids[idx])
            camera_poses.append(cam_pose)
            if C2R is not None:
                T_c = np.eye(4, dtype=np.float64); T_c[:3, :3] = R; T_c[:3, 3] = t
                T_r = C2R @ T_c
                robot_R = T_r[:3, :3]; robot_t = T_r[:3, 3]; robot_r, _ = cv2.Rodrigues(robot_R)
                robot_pose = {"x": float(robot_t[0]), "y": float(robot_t[1]), "z": float(robot_t[2]), "robot_rvec": robot_r.reshape(3).tolist(), "robot_R": robot_R.tolist()}
                if marker_ids and idx < len(marker_ids): robot_pose["id"] = int(marker_ids[idx])
                robot_poses.append(robot_pose)
        return camera_poses, robot_poses

    def plot_aruco(self, image, rotation_matrix = None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        frame_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners)
        rvecs = []; tvecs = [] 
        for i in range(len(corners)):
            rvec_list, tvec_list, _ = self.estimatePoseSingleMarkers(corners[i])
            rvec = np.asarray(rvec_list[0], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec_list[0], dtype=np.float64).reshape(3, 1)
            if rotation_matrix is not None:
                R, _ = cv2.Rodrigues(rvec); R_new = R @ rotation_matrix; rvec, _ = cv2.Rodrigues(R_new)
            frame_markers = cv2.drawFrameAxes(frame_markers, self.camera_intrinsics, np.zeros(5), rvec, tvec, 0.025)
            rvecs.append(rvec); tvecs.append(tvec)
        return rvecs, tvecs, corners, ids, frame_markers

    def aruco_inpainting(self, image, base_image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        rvecs = []; tvecs = []
        for i in range(len(corners)): 
            curr_points = np.array(self.mask_extension(corners[i][0], 0.3, 0.3), dtype = np.int32)
            mask = np.zeros_like(image); cv2.fillConvexPoly(mask, curr_points, (255, 255, 255))
            image = cv2.add(cv2.bitwise_and(image, cv2.bitwise_not(mask)), cv2.bitwise_and(base_image, mask))
            rvec_list, tvec_list, _ = self.estimatePoseSingleMarkers(corners[i])
            rvecs.append(np.asarray(rvec_list[0], dtype=np.float64).reshape(3, 1))
            tvecs.append(np.asarray(tvec_list[0], dtype=np.float64).reshape(3, 1))
        return rvecs, tvecs, corners, ids, image

    def cv2_aruco_inpainting(self, image, inpainting_radius):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(image)
        rvecs = []; tvecs = []
        for i in range(len(corners)):
            curr_points = np.array(self.mask_extension(corners[i][0], self.inpainting_factor, self.inpainting_factor), dtype = np.int32)
            mask = np.zeros_like(image); cv2.fillConvexPoly(mask, curr_points, (255, 255, 255))
            image = cv2.inpaint(image, np.transpose(mask, (2, 0, 1))[0], inpainting_radius, cv2.INPAINT_NS)
            rvec_list, tvec_list, _ = self.estimatePoseSingleMarkers(corners[i])
            rvecs.append(np.asarray(rvec_list[0], dtype=np.float64).reshape(3, 1))
            tvecs.append(np.asarray(tvec_list[0], dtype=np.float64).reshape(3, 1))
        return rvecs, tvecs, corners, ids, image

    def mask_extension(self, corner, extention_x, extention_y):
        vector_x = corner[0] - corner[1]; vector_y = corner[0] - corner[3]
        new_corner = np.zeros_like(corner)
        new_corner[0] = corner[0] + vector_x* extention_x + vector_y* extention_y
        new_corner[1] = corner[1] - vector_x* extention_x + vector_y* extention_y
        new_corner[2] = corner[2] - vector_x* extention_x - vector_y* extention_y
        new_corner[3] = corner[3] + vector_x* extention_x - vector_y* extention_y
        return new_corner
