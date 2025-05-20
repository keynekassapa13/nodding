import cv2
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from nodding.pose import (
    face_mesh,         # MediaPipe FaceMesh instance
    pose,              # MediaPipe Pose instance
    mp_drawing,        # drawing utils
    mp_drawing_styles, # drawing styles
    mp_face,           # face_mesh connections
    mp_pose,           # pose connections
    get_head_pose_angles
)
from nodding.nods import detect_nods

def process_video(input_path, output_dir):
    # Output Paths
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "nod_detection_results.csv")
    out_vid = os.path.join(output_dir, "annotated_video.mp4")
    out_plot = os.path.join(output_dir, "pitch_plot.png")

    logger.info(f"Processing video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logger.error("FPS is zero. Cannot process video.")
        return
    
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_vid, fourcc, fps, (w, h))

    # Extract head pose angles
    pitches, yaws, rolls, timestamps = [], [], [], []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            # pitch = get_pitch(lm, (h, w))
            pitch, yaw, roll = get_head_pose_angles(lm, (h, w))
        else:
            pitch, yaw, roll = np.nan, np.nan, np.nan

        pitches.append(pitch)
        yaws.append(yaw)
        rolls.append(roll)
        timestamps.append(idx / fps)
        idx += 1

    cap.release()
    
    pitches = np.array(pitches)
    yaws   = np.array(yaws)
    rolls  = np.array(rolls)
    
    nod_mask, nod_indices = detect_nods(
        pitches
    )

    # CSV
    with open(out_csv, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["frame", "timestamp_s", "pitch_deg", "yaw_deg", "roll_deg", "is_nod"])
        for i, (t, p, y, r, nod) in enumerate(zip(timestamps, pitches, yaws, rolls, nod_mask)):
            csv_writer.writerow([i, f"{t:.3f}", f"{p:.2f}", f"{y:.2f}", f"{r:.2f}", int(nod)])
    logger.info(f"Saved nod detection CSV → {out_csv}")

    # Annotate video
    cap = cv2.VideoCapture(input_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        final_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        pose_results = pose.process(rgb)

        # Draw Face Mesh
        if face_results.multi_face_landmarks:
            for face_lms in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    final_frame,
                    face_lms,
                    mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style()
                )
        
            # Draw Face Landmarks
            landmark_idxs = [1, 152, 263, 33, 287, 57]
            h, w, _ = final_frame.shape
            for l_idx in landmark_idxs:
                landmark = face_lms.landmark[l_idx]
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(final_frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(final_frame, str(l_idx), (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        # Draw Pose Skeleton
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                final_frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2)
            )

        p = pitches[idx]
        y = yaws[idx]
        r = rolls[idx]

        if not np.isnan(p):
            # text visualisation - (chatgpt) 
            cv2.putText(final_frame, f"Pitch: {p:.1f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(final_frame, f"Yaw:   {y:.1f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(final_frame, f"Roll:  {r:.1f}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        if nod_mask[idx]:
            cv2.putText(final_frame, "NOD!", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        writer.write(final_frame)
        idx += 1

    cap.release()
    writer.release()
    logger.info(f"Saved Annotated video → {out_vid}")

    # Plot pitches
    plt.figure()
    plt.plot(timestamps, pitches, label="Pitch")
    if len(nod_indices):
        plt.scatter(np.array(timestamps)[nod_indices],
                    pitches[nod_indices],
                    color='red', label='Nods')
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()

    logger.info(f"Saved pitch plot → {out_plot}")