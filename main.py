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

def dbfs(x: float) -> float:
    """Convert linear [0..1] to dBFS (cap at very small to avoid -inf)."""
    x = max(1e-12, min(1.0, float(x)))
    return 20.0 * np.log10(x)


def lin_from_db(db: float) -> float:
    """Convert dB to linear (0..1+)."""
    return 10.0 ** (db / 20.0)


class EnvelopeFollower:
    """Peak/RMS hybrid envelope with attack/release smoothing and dB-domain mapping."""
    def __init__(self, sr: float, attack: float = 0.01, release: float = 0.2,
                 low_db: float = -50.0, high_db: float = -10.0):
        self.sr = sr
        self.attack = max(1e-4, attack)
        self.release = max(1e-4, release)
        self.low_db = low_db
        self.high_db = high_db
        self._env = 0.0

        # Precompute coefficients
        self.alpha_a = np.exp(-1.0 / (self.attack * self.sr))
        self.alpha_r = np.exp(-1.0 / (self.release * self.sr))

    def process_block(self, mono_block: np.ndarray) -> float:
        # Use peak of absolute with a light RMS assist to avoid chatter
        abs_block = np.abs(mono_block)
        peak = float(abs_block.max(initial=0.0))
        rms = float(np.sqrt(np.mean(abs_block * abs_block) + 1e-20))
        lvl = 0.7 * peak + 0.3 * rms

        # Attack/Release smoothing in linear domain
        if lvl > self._env:
            self._env = self.alpha_a * self._env + (1.0 - self.alpha_a) * lvl
        else:
            self._env = self.alpha_r * self._env + (1.0 - self.alpha_r) * lvl

        # Map to 0..1 using dB thresholds
        lvl_db = dbfs(self._env)
        t = (lvl_db - self.low_db) / max(1e-6, (self.high_db - self.low_db))
        return float(np.clip(t, 0.0, 1.0))

class EnvelopeFollowerBlockAware:
    """Peak/RMS hybrid envelope with attack/release smoothing and dB-domain mapping.
    Block-aware smoothing so responsiveness doesn't depend on JACK buffer size.
    """
    def __init__(self, sr: float, attack: float = 0.01, release: float = 0.2,
                 low_db: float = -50.0, high_db: float = -10.0):
        self.sr = sr
        self.attack = max(1e-4, attack)
        self.release = max(1e-4, release)
        self.low_db = low_db
        self.high_db = high_db
        self._env = 0.0

    def _alphas_for_frames(self, frames: int):
        # Convert time constants (s) to per-block coefficients given 'frames' samples.
        a = np.exp(-frames / (self.attack * self.sr))
        r = np.exp(-frames / (self.release * self.sr))
        return a, r

    def process_block(self, mono_block: np.ndarray) -> float:
        frames = len(mono_block)
        alpha_a, alpha_r = self._alphas_for_frames(frames)

        # Peak+RMS hybrid to avoid chatter
        abs_block = np.abs(mono_block)
        peak = float(abs_block.max(initial=0.0))
        rms = float(np.sqrt(np.mean(abs_block * abs_block) + 1e-20))
        lvl = 0.7 * peak + 0.3 * rms

        # Attack/Release smoothing in linear domain (per-block)
        if lvl > self._env:
            self._env = alpha_a * self._env + (1.0 - alpha_a) * lvl
        else:
            self._env = alpha_r * self._env + (1.0 - alpha_r) * lvl

        # Map to 0..1 using dB thresholds
        lvl_db = dbfs(self._env)
        t = (lvl_db - self.low_db) / max(1e-6, (self.high_db - self.low_db))
        return float(np.clip(t, 0.0, 1.0))


class AudioModulatedFade:
    def __init__(self, args):
        self.args = args
        self.client = jack.Client("AudioModProcessing")
        self.in_l = self.client.inports.register("in_l")
        self.in_r = self.client.inports.register("in_r")
        self.sr = self.client.samplerate
        self.block_amplitude = 0.0
        self.lock = threading.Lock()
        self.env = EnvelopeFollowerBlockAware(
            sr=self.sr,
            attack=args.attack,
            release=args.release,
            low_db=args.low,
            high_db=args.high,
        )

        # For optional moving average smoothing across a few blocks
        # self.last_vals = deque(maxlen=8)

        @self.client.set_process_callback
        def process(frames: int):
            l = self.in_l.get_array()
            r = self.in_r.get_array()
            ## Convert to mono (mean); arrays are float32 in [-1, 1]
            mono = 0.5 * (l + r)
            # val = self.env.process_block(mono)
            # self.last_vals.append(val)
            # smoothed = float(np.mean(self.last_vals)) if self.last_vals else val
            # with self.lock:
            #     self.block_amplitude = smoothed
            val = self.env.process_block(mono)
            with self.lock:
                self.block_amplitude = val

    def start(self):
        self.client.activate()

    def stop(self):
        try:
            self.client.deactivate()
        finally:
            self.client.close()


def main():
    parser = argparse.ArgumentParser(description="Fade an image by audio amplitude from JACK/PipeWire.")
    parser.add_argument("--image", type=str, default=None, help="Path to image (PNG/JPG). If omitted, use a gradient card.")
    parser.add_argument("--attack", type=float, default=0.005, help="Envelope attack time (s)")
    parser.add_argument("--release", type=float, default=0.000001, help="Envelope release time (s)")
    parser.add_argument("--low", type=float, default=-55.0, help="Threshold low (dBFS)")
    parser.add_argument("--high", type=float, default=-12.0, help="Threshold high (dBFS)")
    parser.add_argument("--fps", type=float, default=60.0, help="Target display FPS")
    parser.add_argument("--width", type=int, default=1980, help="Window width in pixels")
    parser.add_argument("--height", type=int, default=1020, help="Window height in pixels")
    args = parser.parse_args()

    app = AudioModulatedFade(args)

    # Prepare visual background and foreground
    w, h = args.width, args.height

    if args.image is not None:
        img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f"Failed to load image: {args.image}")
        # If image has alpha, drop to BGR for simplicity
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    else:
        # Generate a simple gradient card
        x = np.linspace(0, 1, w, dtype=np.float32)
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        r = (x[None, :] * 255).astype(np.uint8)
        g = (y * 255).astype(np.uint8)
        b = ((1 - x)[None, :] * 255).astype(np.uint8)
        img = np.dstack([b + 0*g, g, r])

    black = np.zeros_like(img)

    app.start()
    cv2.namedWindow("Audio‑Modulated Image processing", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Audio‑Modulated Image processing", w, h)

    frame_interval = 1.0 / max(1.0, args.fps)

    try:
        while True:
            t0 = time.time()
            with app.lock:
                amp = app.block_amplitude  # 0..1

            # Optional easing (smooth response in the GUI loop)
            # amp = np.clip(amp, 0.0, 1.0)

            # Blend foreground (img) over black using amp as alpha
            frame = cv2.addWeighted(img, amp, black, 1.0 - amp, 0.0)

            # HUD overlay
            bar_w = int((w - 40) * amp)
            hud = frame.copy()
            cv2.rectangle(hud, (20, h - 40), (20 + bar_w, h - 20), (255, 255, 255), thickness=-1)
            frame = cv2.addWeighted(hud, 0.35, frame, 0.65, 0)
            cv2.putText(frame, f"amp={amp:.2f}", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Audio‑Modulated Fade", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

            # Simple frame pacing
            dt = time.time() - t0
            if dt < frame_interval:
                time.sleep(frame_interval - dt)
    finally:
        app.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
