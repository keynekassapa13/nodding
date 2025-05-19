import cv2
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

def process_video(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Processing video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video resolution: {w}x{h}, FPS: {fps}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += 1
    cap.release()
    logger.info(f"Total frames: {idx}")
    logger.success("Video processing completed.")
    return frames