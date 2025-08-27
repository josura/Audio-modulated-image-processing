#!/usr/bin/env python3
"""
Audio‑Modulated Fade + Frequency‑Driven Distortion (Python, JACK/PipeWire)
-----------------------------------------------------------------------
"""

import argparse
import threading
import time

import cv2
import jack
import numpy as np


def dbfs(x: float) -> float:
    x = max(1e-12, float(x))
    return 20.0 * np.log10(x)

class EnvelopeFollower:
    """Peak/RMS hybrid envelope with attack/release smoothing and dB-domain mapping.
    Block-aware smoothing so responsiveness doesn't depend on JACK buffer size.

    New parameters:
      - db_offset (dB): shift measured level before mapping
      - db_scale  (scale):  scale measured level in dB before mapping (expand/compress)
      - gamma     (gamma):  nonlinear curve on 0..1 output (post mapping)
      - gate_db   (dB): hard gate; below this, output=0
    """
    def __init__(self, sr: float,
                 attack: float = 0.005, release: float = 0.06,
                 low_db: float = -55.0, high_db: float = -12.0,
                 db_offset: float = 0.0, db_scale: float = 1.0,
                 gamma: float = 1.0, gate_db: float | None = None):
        self.sr = sr
        self.attack = max(1e-4, attack)
        self.release = max(1e-4, release)
        self.low_db = low_db
        self.high_db = high_db
        self.db_offset = db_offset
        self.db_scale = db_scale
        self.gamma = max(1e-4, gamma)
        self.gate_db = gate_db
        self._env = 0.0

    def _alphas_for_frames(self, frames: int):
        a = np.exp(-frames / (self.attack * self.sr))
        r = np.exp(-frames / (self.release * self.sr))
        return a, r

    def process_block(self, mono_block: np.ndarray) -> float:
        frames = len(mono_block)
        alpha_a, alpha_r = self._alphas_for_frames(frames)

        abs_block = np.abs(mono_block)
        peak = float(abs_block.max(initial=0.0))
        rms = float(np.sqrt(np.mean(abs_block * abs_block) + 1e-20))
        lvl = 0.7 * peak + 0.3 * rms

        if lvl > self._env:
            self._env = alpha_a * self._env + (1.0 - alpha_a) * lvl
        else:
            self._env = alpha_r * self._env + (1.0 - alpha_r) * lvl

        lvl_db = dbfs(self._env)
        # Hard gate first
        if (self.gate_db is not None) and (lvl_db < self.gate_db):
            return 0.0
        # Apply offset/scale in dB; e.g., compress or expand sensitivity
        adj_db = (lvl_db + self.db_offset) * self.db_scale
        # Map to 0..1 using thresholds
        t = (adj_db - self.low_db) / max(1e-6, (self.high_db - self.low_db))
        t = float(np.clip(t, 0.0, 1.0))
        # Post-curve (gamma)
        t = t ** self.gamma
        return t


class SpectralBands:
    """Three-band energy tracker with attack/release smoothing (per block)."""
    def __init__(self, sr: float, low_split: float = 200.0, mid_split: float = 2000.0,
                 attack: float = 0.02, release: float = 0.10):
        self.sr = sr
        self.low_split = low_split
        self.mid_split = mid_split
        self.attack = max(1e-4, attack)
        self.release = max(1e-4, release)
        self.state = np.zeros(3, dtype=np.float32)
        self._last_N = None
        self._win = None
        self._freqs = None

    def _ensure_buffers(self, N: int):
        if self._last_N != N:
            self._last_N = N
            self._win = np.hanning(N).astype(np.float32)
            self._freqs = np.fft.rfftfreq(N, d=1.0 / self.sr)

    def _alphas_for_frames(self, frames: int):
        a = np.exp(-frames / (self.attack * self.sr))
        r = np.exp(-frames / (self.release * self.sr))
        return a, r

    def process_block(self, mono_block: np.ndarray) -> np.ndarray:
        N = len(mono_block)
        if N == 0:
            return self.state
        self._ensure_buffers(N)
        alpha_a, alpha_r = self._alphas_for_frames(N)

        x = mono_block * self._win
        spec = np.fft.rfft(x)
        mag = np.abs(spec)
        power = mag * mag
        total = float(np.sum(power) + 1e-20)

        f = self._freqs
        low_mask = f <= self.low_split
        mid_mask = (f > self.low_split) & (f <= self.mid_split)
        high_mask = f > self.mid_split

        bands = np.array([
            float(np.sum(power[low_mask]) / total),
            float(np.sum(power[mid_mask]) / total),
            float(np.sum(power[high_mask]) / total),
        ], dtype=np.float32)

        for i in range(3):
            b = bands[i]
            s = float(self.state[i])
            if b > s:
                s = alpha_a * s + (1.0 - alpha_a) * b
            else:
                s = alpha_r * s + (1.0 - alpha_r) * b
            self.state[i] = s

        return self.state


class AudioModulatedFX:
    def __init__(self, args):
        self.args = args
        self.client = jack.Client("AudioModFade")
        self.in_l = self.client.inports.register("in_l")
        self.in_r = self.client.inports.register("in_r")
        self.sr = self.client.samplerate
        self.lock = threading.Lock()

        self.env = EnvelopeFollower(
            sr=self.sr,
            attack=args.attack,
            release=args.release,
            low_db=args.low,
            high_db=args.high,
            db_offset=args.db_offset,
            db_scale=args.db_scale,
            gamma=args.gamma,
            gate_db=args.gate_db,
        )
        self.spec = SpectralBands(
            sr=self.sr,
            low_split=args.low_split,
            mid_split=args.mid_split,
            attack=args.spect_attack,
            release=args.spect_release,
        )

        self.amp = 0.0
        self.bands = np.zeros(3, dtype=np.float32)

        @self.client.set_process_callback
        def process(frames: int):
            l = self.in_l.get_array()
            r = self.in_r.get_array()
            mono = 0.5 * (l + r)
            amp_val = self.env.process_block(mono)
            bands_val = self.spec.process_block(mono)
            # Optional additional gate for distortion
            if (self.args.dist_gate is not None) and (amp_val < self.args.dist_gate):
                bands_val = bands_val * 0.0
                amp_val = 0.0 if self.args.gate_for_fade else amp_val
            with self.lock:
                self.amp = float(amp_val)
                self.bands[:] = bands_val

    def start(self):
        self.client.activate()

    def stop(self):
        try:
            self.client.deactivate()
        finally:
            self.client.close()


def apply_frequency_distortion(img_bgr: np.ndarray, t: float, amp: float,
                               bands: np.ndarray, scale: float, strength: float,
                               grids_cache: dict) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    key = (w, h)
    g = grids_cache.get(key)
    if g is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        X = (xx / (w - 1) * 2.0) - 1.0
        Y = (yy / (h - 1) * 2.0) - 1.0
        grids_cache[key] = (xx, yy, X, Y)
        xx, yy, X, Y = grids_cache[key]
    else:
        xx, yy, X, Y = g

    low, mid, high = float(bands[0]), float(bands[1]), float(bands[2])

    base = 0.02 * min(w, h)
    A = strength * (0.15 + 0.85 * amp) * (0.3 + 0.7 * low) * base * scale

    kx = 3.0 + 25.0 * mid
    ky = 2.0 + 18.0 * mid

    phase = 2.0 * np.pi * (0.15 * t + 0.35 * high * t)

    dx = A * np.sin(kx * X + phase).astype(np.float32)
    dy = A * np.sin(ky * Y - 1.1 * phase).astype(np.float32)

    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)

    shift = (0.5 + 3.0 * high) * (A / max(1.0, base))

    b, gch, r = cv2.split(img_bgr)

    r_map_x = (map_x + shift).astype(np.float32)
    r_map_y = (map_y + shift * 0.5).astype(np.float32)
    b_map_x = (map_x - shift).astype(np.float32)
    b_map_y = (map_y - shift * 0.5).astype(np.float32)

    r_warp = cv2.remap(r, r_map_x, r_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    g_warp = cv2.remap(gch, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    b_warp = cv2.remap(b, b_map_x, b_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    out = cv2.merge([b_warp, g_warp, r_warp])
    return out


def main():
    parser = argparse.ArgumentParser(description="Audio-modulated fade + frequency-driven distortion")
    parser.add_argument("--image", type=str, default=None, help="Path to image (PNG/JPG). If omitted, use a gradient card.")
    # Envelope
    parser.add_argument("--attack", type=float, default=0.005, help="Envelope attack time (s)")
    parser.add_argument("--release", type=float, default=0.00001, help="Envelope release time (s)")
    parser.add_argument("--low", type=float, default=-55.0, help="Threshold low (dBFS)")
    parser.add_argument("--high", type=float, default=-12.0, help="Threshold high (dBFS)")
    parser.add_argument("--db-offset", type=float, default=-5.0, help="Add this many dB before mapping (positive = louder)")
    parser.add_argument("--db-scale", type=float, default=1.0, help="Scale measured dB before mapping ( >1 more sensitive, <1 less)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Apply output^gamma curve to 0..1 envelope")
    parser.add_argument("--gate-db", type=float, default=None, help="Hard gate in dBFS; below this, envelope = 0")
    # Spectral splits and smoothing
    parser.add_argument("--low-split", type=float, default=200.0, help="Low/Mid split frequency (Hz)")
    parser.add_argument("--mid-split", type=float, default=2000.0, help="Mid/High split frequency (Hz)")
    parser.add_argument("--spect-attack", type=float, default=0.02, help="Spectral bands attack (s)")
    parser.add_argument("--spect-release", type=float, default=0.10, help="Spectral bands release (s)")
    # Distortion controls
    parser.add_argument("--dist-strength", type=float, default=1.0, help="Overall distortion intensity multiplier")
    parser.add_argument("--dist-scale", type=float, default=1.0, help="Spatial scale multiplier for displacement (A)")
    parser.add_argument("--dist-gate", type=float, default=0.02, help="If amp < this, zero the distortion bands (0..1 space)")
    parser.add_argument("--gate-for-fade", action="store_true", help="Also gate the fade when below --dist-gate")
    # UI
    parser.add_argument("--fps", type=float, default=60.0, help="Target display FPS")
    parser.add_argument("--width", type=int, default=1920, help="Window width in pixels")
    parser.add_argument("--height", type=int, default=1080, help="Window height in pixels")
    parser.add_argument(
        "--overlay-info",
        default=True,
        action=argparse.BooleanOptionalAction,   # supports --overlay-info / --no-overlay-info
        help="Show on-screen HUD (amp + band meters). Use --no-overlay-info to hide."
    )
    parser.add_argument(
        "--no-ui-controls",
        default=False,
        action="store_true",
        help="Disable HighGUI controls (no zoom/pan/save UI). Uses AUTOSIZE window."
    )
    
    args = parser.parse_args()

    app = AudioModulatedFX(args)

    w, h = args.width, args.height

    if args.image is not None:
        img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f"Failed to load image: {args.image}")
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    else:
        x = np.linspace(0, 1, w, dtype=np.float32)
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        r = (x[None, :] * 255).astype(np.uint8)
        g = (y * 255).astype(np.uint8)
        b = ((1 - x)[None, :] * 255).astype(np.uint8)
        img = np.dstack([b + 0*g, g, r])

    black = np.zeros_like(img)

    win_name = "Audio‑Modulated FX"
    is_fullscreen = False
    prev_rect = None  # (x, y, w, h) to restore when leaving fullscreen


    app.start()
    if args.no_ui_controls:
        # No interactive UI: no zoom/pan/save toolbar. Window size = frame size.
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        # Do NOT call cv2.resizeWindow in this mode.
    else:
        # Normal resizable window without extra toolbar
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, w, h)

    frame_interval = 1.0 / max(1.0, args.fps)
    t0 = time.time()
    grids_cache = {}

    show_overlay = args.overlay_info
    try:
        while True:
            t = time.time() - t0
            with app.lock:
                amp = float(app.amp)
                bands = app.bands.copy()

            distorted = apply_frequency_distortion(
                img, t=t, amp=amp, bands=bands, scale=args.dist_scale,
                strength=args.dist_strength, grids_cache=grids_cache
            )

            frame = cv2.addWeighted(distorted, amp, black, 1.0 - amp, 0.0)
            # adding HUD overlay
            if show_overlay:
                hud = frame.copy()
                bar_w = int((w - 40) * amp)
                cv2.rectangle(hud, (20, h - 40), (20 + bar_w, h - 20), (255, 255, 255), thickness=-1)
                cv2.putText(
                    hud,
                    f"amp={amp:.2f} low={bands[0]:.2f} mid={bands[1]:.2f} high={bands[2]:.2f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                frame = cv2.addWeighted(hud, 0.35, frame, 0.65, 0)

            cv2.imshow("Audio‑Modulated FX", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('h'), ord('H')):
                show_overlay = not show_overlay
            # else:
            #     print ('you press %s', chr(key))

            # F11; also allow 'f' as a fallback
            if key in (ord('È'), ord('f'), ord('F')):  # È == F11 on my computer it seems
                try:
                    if not is_fullscreen:
                        # Save current rect to restore later
                        try:
                            x, y, ww, hh = cv2.getWindowImageRect(win_name)  # OpenCV >= 4.5
                            prev_rect = (x, y, ww, hh)
                        except Exception:
                            prev_rect = None

                        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        is_fullscreen = True
                    else:
                        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        is_fullscreen = False
                        # Restore size/position if we have it and we're in WINDOW_NORMAL mode
                        if prev_rect is not None and not args.no_ui_controls:
                            x, y, ww, hh = prev_rect
                            try:
                                cv2.resizeWindow(win_name, ww, hh)
                                cv2.moveWindow(win_name, x, y)
                            except Exception:
                                pass
                except Exception:
                    # If the backend doesn’t support fullscreen, just ignore
                    pass

            dt = time.time() - (t0 + t)
            if dt < frame_interval:
                time.sleep(frame_interval - dt)
    finally:
        app.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
