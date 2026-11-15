---
title: Object Tracker
emoji: 🎯
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
short_description: Track any object in your video and measure its speed
---

# Object Tracker

Upload a video, click on the object (or draw a box around it) and get:

- the tracked video with the object's path and box
- speed and distance (in km/h and metres if you mark a reference of known length)
- counts of crossings of a line you draw, and time spent in a zone
- a heatmap of where the object spent time
- CSV / JSON downloads of every position

Powered by [CoTracker3](https://github.com/facebookresearch/co-tracker) (Meta AI).
This free demo runs on CPU, so it processes at most the first 15 seconds of a video
at reduced resolution and takes a few minutes per video.
