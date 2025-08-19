#!/usr/bin/env python3
"""
Audio‑Modulated Fade (Python, JACK/PipeWire)
-------------------------------------------

Reads stereo audio (L/R) from JACK/PipeWire, computes an amplitude envelope,
then uses it to fade an image in/out in real time. This is a minimal first step.

Run:
python3 main.py --image path/to/image.png \
--attack 0.02 --release 0.3 --low -45 --high -6 --fps 60


Press Q or ESC to quit.


Next steps (not implemented here yet): add a visual effect based on top of the fade
(distortion/warp/glitch) whose strength is modulated by the same envelope.
"""


import argparse
import threading
import time
from collections import deque


import cv2
import jack
import numpy as np