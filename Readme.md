# Object_Tracking_in_Self_Recorded_Videos_using_Deep_Learning

##  Overview
This project implements a **user-guided object tracking system** for **self-recorded real-world videos** using a **deep learning-based tracking approach (CoTracker)**.

The system allows a user to select an object of interest in the first frame and **robustly tracks it across subsequent frames**, even under challenging conditions such as:
- camera motion
- illumination changes
- partial occlusion

## Problem Statement
- Input: Self-recorded videos captured using mobile devices or webcams.
- Task: Track a user-specified object (car, face, pet, phone, etc.) across all video frames.
- Model: Utilize CoTracker-v3, a state-of-the-art point tracking model.
- Objective: Extract smooth and accurate trajectories of the selected object.
- Handle occlusions (object temporarily hidden).
- Overcome motion blur during fast movements.
- Maintain stability despite background clutter or noise.
- Expected Outcome: Continuous and reliable motion paths of objects for analysis or visualization.


## Features
- Tracks a user-selected object in self-recorded videos
- Works on real-world videos (not only datasets)
- Handles camera motion and partial occlusion
- Supports different video resolutions and frame rates
- Visualizes tracking results on video frames

## Tech Stack
- Python
- PyTorch
- OpenCV
- NumPy
- CoTracker-v3
- Matplotlib

## Project Structure
```text
object-tracking-project/
├── src/
│   ├── track.py
│   ├── utils.py
│   └── visualize.py
├── models/
│   └── model_loader.py
├── configs/
│   └── config.yaml
├── README.md
├── requirements.txt
└── .gitignore
