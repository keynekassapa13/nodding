import numpy as np
import cv2

# coordinate axes visualisation - (chatgpt) 
def draw_head_orientation(frame, pitch, yaw, roll, center=(150, 200), size=100):
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch),  np.cos(pitch)]])
    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll),  np.cos(roll), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    axes = np.float32([[size,0,0], [0,size,0], [0,0,size]])
    origin = np.float32([[0,0,0]])
    axes_transformed = axes @ R.T + origin
    points_2d = axes_transformed[:, :2] + np.array(center)

    origin_2d = tuple(map(int, center))
    colors = [(0,0,255), (0,255,0), (255,0,0)]
    for pt, color in zip(points_2d, colors):
        pt = tuple(map(int, pt))
        cv2.arrowedLine(frame, origin_2d, pt, color, 2, tipLength=0.2)
