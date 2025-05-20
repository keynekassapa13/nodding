import mediapipe as mp
import numpy as np
import cv2
import math

# Face mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_head_pose_angles(landmarks, image_shape):
    """
    Source: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    Args:
        landmarks(list): List of facial landmarks.
        image_shape(tuple): Shape of the image (height, width).
    Returns:
        tuple: Pitch, Yaw, Roll angles in degrees.
    """
    h, w = image_shape
    # Model Points / Object Points
    model_points = np.array([
        [   0.0,    0.0,     0.0],     # nose tip
        [   0.0, -330.0,   -65.0],     # chin
        [-225.0,  170.0,  -135.0],     # left eye outer
        [ 225.0,  170.0,  -135.0],     # right eye outer
        [-150.0, -150.0,  -125.0],     # left mouth corner
        [ 150.0, -150.0,  -125.0],     # right mouth corner
    ], dtype=np.float64)

    # Image Points
    idxs = [1, 152, 263, 33, 287, 57] 
    # tip, chin, right eye outer, left eye outer, right mouth corner, left mouth corner
    image_points = np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in idxs
    ], dtype=np.float64)

    # Camera internals
    focal_length = w
    cam_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    # assume no lens distortion
    dist_coeffs = np.zeros((4,1)) 

    # Solve PnP. pose estimation
    success, rot_vec, trans_vec = cv2.solvePnP(
        model_points, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return np.nan, np.nan, np.nan

    # rotation matrix (chatgpt)
    rot_mat, _ = cv2.Rodrigues(rot_vec)

    # full projection matrix, then decompose (chatgpt)
    proj_mat = cam_matrix.dot(np.hstack((rot_mat, trans_vec)))
    (cam_matrix2,
    rot_matrix2,
    trans_vect2,
    rotX,
    rotY,
    rotZ,
    euler_angles) = cv2.decomposeProjectionMatrix(proj_mat)
    pitch, yaw, roll = euler_angles.flatten()
    return float(pitch), float(yaw), float(roll)