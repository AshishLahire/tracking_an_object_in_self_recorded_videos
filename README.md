<div align="center">

# 🎯 Object Tracking in Self-Recorded Videos

**Click on anything in your own video, and measure how it moves.**

Track an object through a phone or camera video, then measure its speed, count line crossings,
measure joint angles for rehab, or measure gravity for a physics lab.
Built on [CoTracker3](https://github.com/facebookresearch/co-tracker), PyTorch and OpenCV.

[![Tests](https://github.com/AshishLahire/tracking_an_object_in_self_recorded_videos/actions/workflows/tests.yml/badge.svg)](https://github.com/AshishLahire/tracking_an_object_in_self_recorded_videos/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AshishLahire/tracking_an_object_in_self_recorded_videos/blob/main/notebooks/colab_demo.ipynb)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Gradio](https://img.shields.io/badge/Web%20app-Gradio-F97316)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

<img src="docs/gifs/plane_speed.gif" width="520" alt="Plane takeoff tracked with a moving box and live speed">

<sub>A Boeing 737 takeoff filmed with a panning camera. Using the plane's own 39.5 m length as the ruler and
compensating for the camera's motion, the measured speed reaches about 290 km/h at liftoff,
in line with the 737-800's typical takeoff speed (about 280–300 km/h).</sub>

[Features](#-features) •
[Demo](#-demo) •
[Accuracy](#-accuracy) •
[Quick start](#-quick-start) •
[Usage](#-usage) •
[Deploy](#-deploy) •
[How it works](#-how-it-works)

</div>

---

## ✨ Features

### 📍 Tracking
- **Click to track:** select one or more points, or draw a box. Box mode tracks a grid of points, ignores background caught in the box, and moves the box with the object.
- **Built for real phone videos:** handles occlusion (the path breaks while the object is hidden), any resolution (large videos are downscaled as they load), and any frame rate (kept in the output).
- **Camera-shake compensation:** removes handheld shake and panning, so measurements reflect the object's own motion.

### 📏 Measurement
- **Speed and distance** in km/h and metres, from a reference of known length. If the reference is the object itself, the scale follows its size as it moves closer or farther away.
- **Line crossings** with time and direction, e.g. counting vehicles or laps.
- **Zones:** entries, exits and time spent inside a region.
- **Heatmap** of where the object spent its time.

### 🚦 Several objects and a two-line speed trap
Draw a box around each object (cars, players, animals). Each one gets its own colour, path, speed and line crossings, all from a single tracking run.

For speed, draw **line A** and **line B** across the road and enter the real distance between them. Each object's speed is the distance ÷ the time it takes to get from one line to the other, with crossing times interpolated between frames. This is how average-speed cameras work, and it doesn't depend on the camera angle. Passes over a speed limit are flagged.

### 🦵 Joint angles for physiotherapy and rehab
Click three points (e.g. hip → knee → ankle) to get the joint angle in every frame, the **range of
motion** and the **number of repetitions**. Range of motion is normally measured by hand with a goniometer;
with this, a patient can film an exercise at home and compare progress week by week.

### 🧪 Physics lab
- **Projectile or falling object:** measures **g**, launch speed and launch angle. The throw and the landing are trimmed off automatically.
- **Pendulum:** measures the **period**, and g from the pendulum's length.

It turns a phone into lab equipment for schools that have no motion sensors.

### 📦 Outputs
- An annotated **H.264 video** that plays in any browser.
- **CSV files:** one row per point, one row per frame for the object, and joint angles.
- A **JSON summary** and a **heatmap PNG**.

All of it is available from a **web app**, the **command line** and a **Python API**.

---

## 🎬 Demo

| Speed trap: two cars | Joint angle and repetitions | Projectile: g from a throw |
|:---:|:---:|:---:|
| <img src="docs/gifs/speed_trap.gif" width="240"> | <img src="docs/gifs/joint_angle.gif" width="240"> | <img src="docs/gifs/physics_projectile.gif" width="240"> |
| **Pendulum: period and g** | | |
| <img src="docs/gifs/physics_pendulum.gif" width="240"> | | |

| Bicycle | Car (drone shot) | Plane |
|:---:|:---:|:---:|
| <img src="docs/gifs/bicycle.gif" width="240"> | <img src="docs/gifs/car_side.gif" width="240"> | <img src="docs/gifs/plane.gif" width="240"> |
| **Tiger** | **Paragliding** | **Apple (moving camera)** |
| <img src="docs/gifs/tiger.gif" width="240"> | <img src="docs/gifs/paragliding.gif" width="240"> | <img src="docs/gifs/apple.gif" width="240"> |

<sub>The joint-angle and physics clips are rendered test videos with known ground truth, used to measure accuracy (see below).</sub>

---

## 📊 Accuracy

Each measurement mode was checked with the real model against videos where the true answer is known.

| Test | Measured | True value |
|---|---|---|
| Speed trap, two cars in opposite lanes (15 m trap) | **50.02** and **72.03 km/h** (+0.04%); the faster car flagged over 60 km/h | 50 and 72 km/h |
| Joint angle, hinge bending 170° → 90° (3 reps) | 0.5° mean error (1.2° max); ROM 78.4°; 3 reps | ROM 79.6°; 3 reps |
| Projectile: gravity | **9.817 ± 0.075 m/s²** (+0.1%) | 9.81 m/s² |
| Projectile: launch speed and angle | 2.89 m/s at 58.7° | 2.92 m/s at 59.0° |
| Pendulum period (0.6 m) | **1.5541 s** → g = 9.808 m/s² (+0.01%) | 1.5539 s |
| Plane takeoff (real video) | about 290 km/h at liftoff | 737-800: about 280–300 km/h |
| Lower model resolution (192×256 vs 384×512) | tracked points within about 1 px | — |

The test videos are clean, rendered scenes, so expect lower accuracy on real footage. See [Limitations](#%EF%B8%8F-limitations).

---

## 🚀 Quick start

```bash
git clone https://github.com/AshishLahire/tracking_an_object_in_self_recorded_videos.git
cd tracking_an_object_in_self_recorded_videos
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:7860**.

The first run downloads the CoTracker3 model (about 100 MB). A GPU is optional: on a CPU, a 10-second clip takes a few minutes.

**No install:** open the [Colab notebook](https://colab.research.google.com/github/AshishLahire/tracking_an_object_in_self_recorded_videos/blob/main/notebooks/colab_demo.ipynb). It runs the app on a free GPU and gives you a temporary public link.

---

## 🧭 Usage

### Web app
1. Upload a video, or pick an example.
2. Choose the frame where the object is clearly visible.
3. Choose what a click adds, then click on the frame:
   - **Point** or **Box:** the object to track. Draw several boxes to track several objects.
   - **Speed trap lines:** line A and line B across the path, then enter the distance between them.
   - **Joint angle:** three clicks: end → joint → end.
   - **Reference length**, **Counting line** or **Zone**.
4. Optionally open **Physics lab** and choose *Projectile* or *Pendulum*.
5. Press **Track**. You get the video, a summary, charts and downloads.

### Command line

```bash
# track one point on the cyclist, selected on frame 70
python cli.py --video examples/bicycle.mp4 --frame 70 --point 125,50
```

```bash
# vehicle speed: box + the plane's own length as the ruler + camera-motion compensation
python cli.py --video examples/plane.mp4 --box 205,100,372,152 --grid 5 --stabilize \
    --reference 210,135,345,127,39.5 --reference-is-object --csv results/plane.csv
```

```bash
# rehab: knee range of motion and repetitions (hip, knee, ankle on frame 0, filmed side-on)
python cli.py --video my_exercise.mp4 --joint 210,120,220,260,230,400 --joint-name knee --csv results/knee.csv
```

```bash
# physics lab: g from a thrown ball, with a 1 m ruler in the same plane
python cli.py --video throw.mp4 --point 120,300 --physics projectile --reference 100,340,350,340,1.0
```

```bash
# physics lab: pendulum period, and g from its 0.6 m length
python cli.py --video pendulum.mp4 --point 335,198 --physics pendulum --pendulum-length 0.6
```

```bash
# speed trap: two cars, lines A and B 15 m apart, limit 60 km/h
python cli.py --video road.mp4 --object 20,170,110,210 --object 560,235,640,275 \n    --object-name "car 1" --object-name "car 2" \n    --trap 150,140,150,300,450,140,450,300,15 --speed-limit 60 --csv results/road.csv
```

```bash
# count line crossings and time spent in a zone
python cli.py --video examples/bicycle.mp4 --frame 70 --point 125,50 \
    --line 300,60,300,200 --zone "250,100;420,100;420,240;250,240"
```

All coordinates are pixels of the original video, on `--frame`. Run `python cli.py --help` for every option.

<details>
<summary><b>Python API</b></summary>

```python
from tracker import track_video   # with src/ on the path, or after `pip install -e .`

r = track_video("examples/plane.mp4", box=(205, 100, 372, 152), box_grid=5, stabilize=True,
                reference=(210, 135, 345, 127, 39.5), reference_is_object=True,
                output_path="results/plane.mp4", csv_path="results/plane.csv")
print(r.summary()["motion"])          # distance, average and top speed

r = track_video("my_exercise.mp4", joints=[[(210, 120), (220, 260), (230, 400)]], joint_names=["knee"])
print(r.joints[0]["rom_deg"], r.joints[0]["reps"])
```
</details>

<details>
<summary><b>GPU vs CPU (<code>--mode</code>, <code>--model-res</code>)</b></summary>

- **`offline`** sees the whole video at once. It gives the best quality and also tracks before the selected frame, but needs a GPU.
- **`online`** works through 16-frame windows, so memory stays flat. It tracks forward only.
- **`auto`** (the default) uses offline on a GPU and online on a CPU.
- **`--model-res 192x256`** roughly halves memory use and doubles speed, with points within about 1 px of full resolution.
</details>

---

## 🌍 Real-world uses

| Who | Problem | What this gives them |
|---|---|---|
| 🦵 Physiotherapists and patients | Range of motion is measured by hand, only in the clinic | Joint angle, range of motion and reps from a home video |
| 🧪 Students and teachers | No motion sensors or photogates in many schools | g, launch speed and pendulum period from a phone video |
| 🚗 Residents and road-safety studies | Proving speeding needs a radar gun or a traffic survey | A phone on a tripod plus a two-line speed trap: speed per vehicle, share over the limit, CSV as evidence |
| ⚽ Coaches and athletes | Ball and sprint speed, paths | Point or box tracking, speed, CSV export |
| 🐾 Wildlife researchers | Movement patterns from field footage | Box tracking, heatmap, zones |

---

## 🌐 Deploy

| Option | Cost | Notes |
|---|---|---|
| **Google Colab** | Free | One click from the badge above. Free GPU, temporary public link. |
| **Render** (`render.yaml` + `Dockerfile`) | Paid (2 GB plan) | Dashboard → *New → Blueprint* → select this repo. The model peaked at about 1.8 GB in testing with the defaults below, so 512 MB plans run out of memory. |
| **Hugging Face Spaces** | Free (16 GB RAM) | `python deploy/prepare_space.py`, then upload `build/space` as a Gradio Space. |
| **Any Docker host** | — | `docker build -t object-tracker .` then `docker run -p 10000:10000 object-tracker` |

**Settings for small servers** (set them as environment variables):
- `TRACKER_MODEL_RES=192x256`: lower model resolution, about half the memory.
- `MAX_SECONDS`: longest part of a video that is processed.
- `MAX_SIDE`: frames are downscaled to this longer side.

---

## 🔍 How it works

```mermaid
flowchart LR
    A[Video] --> B[Load and resize<br/>frame by frame]
    B --> C[CoTracker3<br/>point tracking]
    S[Points / box / joints<br/>clicked on one frame] --> C
    C --> D[Occlusion-aware<br/>smoothing]
    D --> E[Object box<br/>background filtering]
    E --> F[Camera-motion<br/>compensation]
    F --> G{Measurements}
    G --> H[Speed, distance,<br/>lines, zones, heatmap]
    G --> I[Joint angles,<br/>ROM, repetitions]
    G --> J[Physics fits:<br/>projectile, pendulum]
    H & I & J --> K[Video, CSV, JSON]
```

1. **Tracking.** CoTracker3 follows the selected points through every frame and predicts when each point is hidden.
2. **Cleaning.** Paths are smoothed only across visible stretches. In box mode, points that don't move with the object (background caught in the box) are dropped.
3. **Measuring.** Pixels become metres through a reference of known length. Camera motion is estimated from the background with sparse optical flow and removed.
4. **Models.** Joint angles come from vector geometry. Physics uses least-squares fits of `y = y₀ + v₀t + ½gt²` and of a damped cosine, with standard errors.

---

## 📁 Project structure

```
├── app.py                  # web app (Gradio)
├── cli.py                  # command line
├── src/tracker/
│   ├── pipeline.py         # track_video(): video + selection -> video, CSV, JSON, measurements
│   ├── model.py            # CoTracker3 (offline / online modes, model resolution)
│   ├── selection.py        # box -> points, points -> moving box, background filtering
│   ├── analysis.py         # speed, distance, line crossings, zones, heatmap
│   ├── multi.py            # several objects in one run
│   ├── speedtrap.py        # two-line speed trap with sub-frame timing
│   ├── joints.py           # joint angles, range of motion, repetition counting
│   ├── physics.py          # projectile and pendulum fits
│   ├── stabilize.py        # camera-motion compensation
│   ├── smoothing.py        # occlusion-aware smoothing
│   ├── draw.py             # overlays: paths, boxes, joints, curves, labels
│   ├── video_io.py         # video read/write, CSV export, H.264 re-encode, GIFs
│   ├── metrics.py          # tracking metrics
│   └── cli.py
├── notebooks/              # demo, Colab and the original experiment
├── examples/               # sample videos
├── results/videos/         # tracked results
├── docs/                   # project report, screenshots, GIFs
├── tests/                  # 68 tests, no model download needed
├── deploy/                 # Hugging Face Space packaging
├── Dockerfile · render.yaml
└── requirements.txt · requirements-dev.txt · pyproject.toml
```

Run the tests:

```bash
pip install -r requirements-dev.txt
```

```bash
python -m pytest
```

---

## ⚠️ Limitations

- **Recorded video only.** This works on recorded clips, not live CCTV. A CPU takes about 1 minute per second of video at full quality.
- **Measurements are 2D.** For accurate speed, angles and physics, film side-on, with the reference at the same distance as the object.
- **Joint angles** are measured in the video plane. Loose clothing over the joint lowers accuracy. Use them as a progress aid, not a medical diagnosis.
- **Physics lab** needs a still camera. The pendulum formula assumes small swings (under about 15°). Higher frame rates (60 fps) give better results.
- **Consent.** Do not track people without their consent.

---

## 🙏 Credits

- **[CoTracker3](https://github.com/facebookresearch/co-tracker)** by Meta AI: the point-tracking model. Its code and weights are licensed **CC BY-NC 4.0**, which allows **non-commercial use only**.
- Developed by **Ashish Lahire**. The project started as a research task at **IIT Mandi**. The original project report is in [`docs/report/`](docs/report/Project_Report.pdf).

<div align="center">

⭐ If this project is useful to you, please star the repository.

</div>
