import os, time, math, random, urllib.request
from dataclasses import dataclass

import numpy as np
import cv2

import gradio as gr
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


@dataclass
class P:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    life: float
    age: float
    col: tuple


class Universe:
    def __init__(self, w=960, h=540, maxp=1400):
        self.w, self.h = w, h
        self.cx, self.cy = w / 2, h / 2
        self.maxp = maxp
        self.particles = []
        self.last = time.time()
        self.expansion = 0.0
        self.blackhole = 0.0
        self.create = 0.0
        self.swirl = 0.0
        self.spawn_acc = 0.0

    def spawn(self, n):
        palette = [
            (255, 255, 255),
            (255, 240, 200),
            (200, 220, 255),
            (255, 210, 210),
            (210, 255, 230),
        ]
        for _ in range(n):
            if len(self.particles) >= self.maxp:
                return
            ang = random.random() * math.tau
            rr = (random.random() ** 1.8) * 18.0
            x = self.cx + math.cos(ang) * rr
            y = self.cy + math.sin(ang) * rr
            speed = 90 + 220 * random.random()
            vx = math.cos(ang) * speed + random.uniform(-40, 40)
            vy = math.sin(ang) * speed + random.uniform(-40, 40)
            self.particles.append(
                P(x=x, y=y, vx=vx, vy=vy, r=random.uniform(1.0, 2.6),
                  life=random.uniform(3.0, 7.0), age=0.0, col=random.choice(palette))
            )

    def step(self, dt):
        # creation -> spawn
        base_rate = 35.0
        burst_rate = 450.0
        rate = base_rate + burst_rate * self.create
        self.spawn_acc += rate * dt
        n = int(self.spawn_acc)
        self.spawn_acc -= n
        self.spawn(min(n, 60))

        alive = []
        for p in self.particles:
            p.age += dt
            if p.age >= p.life:
                continue

            dx = p.x - self.cx
            dy = p.y - self.cy
            r = math.hypot(dx, dy) + 1e-6
            ux, uy = dx / r, dy / r

            # expansion/collapse
            radial_k = 200.0
            a_rad = self.expansion * radial_k / (1.0 + (r / 180.0) ** 2)

            # blackhole
            bh_k = 850.0
            a_bh = self.blackhole * bh_k / (1.0 + (r / 120.0) ** 2)

            ax = (a_rad - a_bh) * ux
            ay = (a_rad - a_bh) * uy

            # swirl
            swirl_k = 140.0
            ax += self.swirl * swirl_k * (-uy) / (1.0 + r / 250.0)
            ay += self.swirl * swirl_k * (ux) / (1.0 + r / 250.0)

            damp = 0.995
            p.vx = (p.vx + ax * dt) * damp
            p.vy = (p.vy + ay * dt) * damp
            p.x += p.vx * dt
            p.y += p.vy * dt

            if self.blackhole > 0.15 and r < 16:
                continue

            alive.append(p)

        self.particles = alive

    def render(self):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # core glow
        core_r = int(8 + 28 * self.blackhole + 12 * max(0.0, self.expansion))
        cv2.circle(img, (int(self.cx), int(self.cy)), core_r, (70, 30, 120), -1)
        cv2.circle(img, (int(self.cx), int(self.cy)), max(6, core_r // 2), (160, 110, 220), 2)

        for p in self.particles:
            t = clamp(p.age / p.life, 0.0, 1.0)
            fade = int(255 * (1.0 - t))
            col = (p.col[0] * fade // 255, p.col[1] * fade // 255, p.col[2] * fade // 255)
            cv2.circle(img, (int(p.x), int(p.y)), max(1, int(p.r)), col, -1)

        # slight bloom
        blur = cv2.GaussianBlur(img, (0, 0), 2.0)
        out = cv2.addWeighted(img, 0.85, blur, 0.35, 0)
        return out


ensure_model()

# MediaPipe landmarker
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
landmarker = vision.HandLandmarker.create_from_options(options)

U = Universe()
_ts = 0


def controls_from_landmarks(hand, w, h):
    def pt(i):
        lm = hand[i]
        return lm.x * w, lm.y * h

    wx, wy = pt(0)
    mx, my = pt(9)
    ixm, iym = pt(5)
    tx, ty = pt(4)
    ix, iy = pt(8)
    midx, midy = pt(12)

    palm_ref = 0.5 * (math.hypot(mx - wx, my - wy) + math.hypot(ixm - wx, iym - wy))
    palm_ref = max(palm_ref, 30.0)

    d_ti = math.hypot(ix - tx, iy - ty)
    d_tm = math.hypot(midx - tx, midy - ty)

    pinch_ti = d_ti / palm_ref
    pinch_tm = d_tm / palm_ref

    expansion = np.interp(pinch_ti, [0.20, 1.00], [-1.0, 1.0])
    expansion = float(clamp(expansion, -1.0, 1.0))

    blackhole = float(clamp((0.28 - pinch_ti) / 0.28, 0.0, 1.0))
    create = float(clamp((0.30 - pinch_tm) / 0.30, 0.0, 1.0))

    ang = math.atan2(my - wy, mx - wx)
    swirl = float(clamp(math.sin(ang), -1.0, 1.0))

    return expansion, blackhole, create, swirl


def step(webcam_frame):
    """
    webcam_frame: RGB numpy from Gradio
    returns RGB image
    """
    global _ts

    now = time.time()
    dt = now - U.last
    U.last = now
    dt = float(clamp(dt, 1/120, 1/10))

    # resize incoming frame for speed
    frame = webcam_frame
    if frame is None:
        U.step(dt)
        out = U.render()
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    # Gradio gives RGB; mediapipe expects SRGB (RGB)
    h, w, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    _ts += int(dt * 1000)
    result = landmarker.detect_for_video(mp_image, _ts)

    # smooth
    target_exp, target_bh, target_cr, target_sw = 0.0, 0.0, 0.0, 0.0
    if result.hand_landmarks:
        target_exp, target_bh, target_cr, target_sw = controls_from_landmarks(result.hand_landmarks[0], w, h)

    s = 0.18
    U.expansion = U.expansion + (target_exp - U.expansion) * s
    U.blackhole = U.blackhole + (target_bh - U.blackhole) * s
    U.create = U.create + (target_cr - U.create) * s
    U.swirl = U.swirl + (target_sw - U.swirl) * s

    U.step(dt)
    out_bgr = U.render()
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb


with gr.Blocks() as demo:
    gr.Markdown("# Hand Universe (Python hosted)")
    gr.Markdown(
        "Thumb-Index open: Expansion | Thumb-Index pinch: Collapse/Blackhole | Thumb-Middle pinch: Creation\n\n"
        "Note: This runs on a server; webcam frames are sent for processing."
    )
    inp = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
    out = gr.Image(type="numpy", label="Universe Output")
    inp.stream(step, inputs=inp, outputs=out)

demo.launch()
