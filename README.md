# Football Analysis Project

## Introduction
The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

![Screenshot](output_videos/screenshot.png)

## Modules Used
The following modules are used in this project:
- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Trained Models
- [Trained Yolo v5](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Sample video
-  [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Requirements
To run this project, you need the following:
- Python 3.9 or newer (virtual environments recommended)
- The Python dependencies listed in `requirements.txt`
- A trained YOLOv5/YOLOv8 checkpoint (see below)

Install the dependencies with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Installing `ultralytics` will also pull in PyTorch. If you need GPU acceleration, follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for the wheel that matches your CUDA version, then rerun `pip install -r requirements.txt`.

## Project setup

1. **Download the model weights**
	- Grab `best.pt` from the link above and place it in `models/best.pt`.
2. **Provide an input video**
	- Copy a match video into `input_videos/` (e.g. `input_videos/08fd33_4.mp4`).
3. **(Optional) Use provided stubs**
	- The `stubs/` directory contains cached detections and camera movement values so you can generate an output video quickly without running the full detector on every frame. Keep these files in place for a fast first run.

## Running the pipeline

After completing the setup steps and activating your environment, run:

```powershell
python main.py
```

The annotated video will be saved to `output_videos/output_video.avi`. Intermediate artefacts such as cropped images and screenshots are also written to `output_videos/` for convenience.

### Quick YOLO inference demo

If you only want to run YOLO on a video and inspect the raw detections, execute:

```powershell
python yolo_inference.py
```

This will write prediction frames to the `runs/` directory created by Ultralytics and echo detection metadata to the console.