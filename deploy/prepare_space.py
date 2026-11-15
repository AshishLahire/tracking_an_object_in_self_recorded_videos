"""Assemble a ready-to-upload Hugging Face Space in build/space/.

    python deploy/prepare_space.py
    huggingface-cli login                       # once, with a write token from huggingface.co/settings/tokens
    huggingface-cli upload <your-username>/object-tracker build/space . --repo-type=space

The Space runs app.py on free CPU hardware. Limits are set through environment variables
in the generated README / app (MAX_SECONDS, MAX_SIDE); change them in the Space settings.
"""

import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "build", "space")

# Short clips keep the Space small and the demo fast on CPU
EXAMPLES = ["plane.mp4", "apple.mp4", "car_side.mp4", "paragliding.mp4"]

SPACE_REQUIREMENTS = """\
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.0
opencv-python-headless>=4.8
numpy>=1.24
scipy>=1.10
pandas>=1.5
gradio>=5,<6
imageio-ffmpeg>=0.5
"""

# Free CPU Spaces are slow: process less video, at lower resolution
APP_PREFIX = """\
import os
os.environ.setdefault("MAX_SECONDS", "15")
os.environ.setdefault("MAX_SIDE", "512")
os.environ.setdefault("HOST", "0.0.0.0")
"""


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "examples"))

    shutil.copytree(os.path.join(ROOT, "src", "tracker"), os.path.join(OUT, "src", "tracker"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    with open(os.path.join(ROOT, "app.py"), encoding="utf-8") as f:
        app = f.read()
    with open(os.path.join(OUT, "app.py"), "w", encoding="utf-8") as f:
        f.write(APP_PREFIX + app)
    for name in EXAMPLES:
        shutil.copy2(os.path.join(ROOT, "examples", name), os.path.join(OUT, "examples", name))
    shutil.copy2(os.path.join(ROOT, "deploy", "huggingface", "README.md"), os.path.join(OUT, "README.md"))
    with open(os.path.join(OUT, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(SPACE_REQUIREMENTS)

    size = sum(os.path.getsize(os.path.join(d, n)) for d, _, files in os.walk(OUT) for n in files)
    print(f"Space ready in {OUT} ({size / 1e6:.1f} MB)")
    print("Upload with:\n  huggingface-cli upload <your-username>/object-tracker build/space . --repo-type=space")


if __name__ == "__main__":
    main()
