"""Run the tracker from the project folder without installing it.

    python cli.py --video examples/bicycle.mp4 --frame 70 --point 125,50

(After `pip install -e .` the same command is available as `object-tracker ...`.)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from tracker.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
