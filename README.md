# Nodding

## Introduction

Detect head nodding behavior (up-and-down movement) from a single video file using MediaPipe face landmarks and head pose estimation.

- Head pose estimation via MediaPipe face mesh
- Pitch/Yaw/Roll calculation using `solvePnP` and `cv2.decomposeProjectionMatrix`
- Robust nod detection using:
  - Smoothed pitch angle
  - Hysteresis thresholding
  - Yaw/Roll filtering
- Return:
  - Annotated video (`.mp4`)
  - CSV results
  - Pitch plot with nod markers
- Assumption:
  - Camera
    - The focal length equals the image width and the principal point is at the image center
    - No lens distortions from camera
  - Video
    - Single face in a video and the person is facing to the camera
  - Nods
    - All nods follow the nod_template
    - Nods will be contained within a fixed-size window 

## Installation

***Any configuration can be changed in config.json***

```bash
pip install -e .
```

or 

```bash
pip install git+https://github.com/keynekassapa13/nodding.git
```

If needed, install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Once installed, use the CLI:

```bash
nodding --input data/input/vid.mp4 --output data/output/
```

## Output

After running, you will find:

```bash
data/output/
├── annotated_video.mp4
├── nod_detection_results.csv
└── pitch_plot.png
```

## Packaging (for maintaners)

Build and install 

```bash
pip install -e .
```