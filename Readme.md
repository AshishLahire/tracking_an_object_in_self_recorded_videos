# Object_Tracking_in_Self_Recorded_Videos_using_Deep_Learning

## 🔍 Project Overview

This repository implements a robust object tracking system that processes self-recorded videos to follow a user-selected object throughout the video. The system uses a deep learning-based tracker (CoTracker-v3) combined with OpenCV and PyTorch to handle real-world video challenges such as motion, partial occlusion, noise, and varying frame rates.

This project was developed as part of a research/task at IIT Mandi.

---

## 🎯 Goal

Given a self-recorded input video and a selected object in the first frame, the system:

✔ Tracks the chosen object across all frames  
✔ Handles motion and occlusion  
✔ Works with different video formats and resolutions  
✔ Visualizes the tracked path on video frames

---

## 💡 Key Features

✔ Supports self-recorded videos (mobile, webcam, etc.)  
✔ Uses deep learning model **CoTracker-v3** for accurate tracking  
✔ Real-time object tracking and visualization  
✔ Handles environmental challenges like motion blur and background clutter  
✔ Shows tracked trajectory on output video or frames

---

## 🧠 Tech Stack

- **Python** – Main programming language  
- **PyTorch** – Deep learning framework  
- **OpenCV** – Video processing and visualization  
- **NumPy** – Numeric operations  
- **CoTracker-v3** – Object tracking model  
- **Matplotlib** – Visualization (optional)

---

📁 Project Structure
tracking_an_object_in_self_recorded_videos/


├── input/
    
   └── (User-uploaded or sample videos)

├── Output_final/
    
   └── (Generated result videos/images)

├── Report/
    
  └── (Project documentation and report files)

├── main.ipynb
    
  └── (Main execution notebook for object tracking)

├── requirements.txt
    
│   └── (Python dependencies)


├── README.md
    
   └── (Project description and instructions)

└── .gitignore
    
    └── (Ignored files for version control)


