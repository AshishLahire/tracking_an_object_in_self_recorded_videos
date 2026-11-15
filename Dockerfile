# Web app image (used by Render, works on any Docker host).
#   docker build -t object-tracker .
#   docker run -p 7860:7860 -e PORT=7860 object-tracker
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/.torch

# runtime libraries OpenCV (headless) needs
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image ~1.5 GB smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install "opencv-python-headless>=4.8" "numpy>=1.24" "scipy>=1.10" "pandas>=1.5" \
       "gradio>=5,<6" "imageio-ffmpeg>=0.5"

# Download the CoTracker3 online model at build time, so the app starts without a download
RUN python -c "import torch; torch.hub.load('facebookresearch/co-tracker', 'cotracker3_online')"

COPY src ./src
COPY app.py ./
COPY examples/plane.mp4 examples/apple.mp4 examples/car_side.mp4 examples/paragliding.mp4 ./examples/

# Small-instance defaults (override in the Render dashboard):
#   TRACKER_MODEL_RES 192x256 -> ~1.8 GB peak in testing (vs ~3.5 GB at full resolution): needs a 2 GB instance
ENV HOST=0.0.0.0 \
    PORT=10000 \
    MAX_SECONDS=10 \
    MAX_SIDE=384 \
    TRACKER_MODEL_RES=192x256

EXPOSE 10000
CMD ["python", "app.py"]
