import os
import sys
import math
import random
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pygame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: tuple[int, int, int]
    life: float  # seconds
    age: float = 0.0

    def update(self, dt: float, cx: float, cy: float, expansion: float, swirl: float, blackhole: float) -> bool:
        """
        Returns True if still alive, False if should be removed.
        """
        self.age += dt
        if self.age >= self.life:
            return False

        dx = self.x - cx
        dy = self.y - cy
        r = math.hypot(dx, dy) + 1e-6

        # Unit vectors
        ux = dx / r
        uy = dy / r

        # Expansion (+) or collapse (-): radial acceleration
        # Make it stronger near center for dramatic effect.
        radial_k = 220.0
        a_rad = expansion * radial_k / (1.0 + (r / 180.0) ** 2)

        # Black hole: strong inward pull + deletion near center
        bh_k = 900.0
        a_bh = blackhole * bh_k / (1.0 + (r / 120.0) ** 2)

        ax = (a_rad * ux) - (a_bh * ux)
        ay = (a_rad * uy) - (a_bh * uy)

        # Swirl: perpendicular acceleration to create galaxy rotation
        # Perp unit vector ( -uy, ux )
        swirl_k = 160.0
        ax += swirl * swirl_k * (-uy) / (1.0 + r / 250.0)
        ay += swirl * swirl_k * (ux) / (1.0 + r / 250.0)

        # Mild damping for stability
        damp = 0.995
        self.vx = (self.vx + ax * dt) * damp
        self.vy = (self.vy + ay * dt) * damp

        self.x += self.vx * dt
        self.y += self.vy * dt

        # If blackhole active and particle very close, "destroy" it
        if blackhole > 0.15 and r < 18.0:
            return False

        return True

    def draw(self, screen: pygame.Surface) -> None:
        # Fade out with age
        t = clamp(self.age / self.life, 0.0, 1.0)
        fade = int(lerp(255, 0, t))
        c = (min(255, self.color[0] * fade // 255),
             min(255, self.color[1] * fade // 255),
             min(255, self.color[2] * fade // 255))
        pygame.draw.circle(screen, c, (int(self.x), int(self.y)), max(1, int(self.radius)))


def spawn_particle(cx: float, cy: float, base_speed: float, spread: float) -> Particle:
    # Start near center with random direction
    angle = random.random() * math.tau
    r = (random.random() ** 1.8) * 18.0
    x = cx + math.cos(angle) * r
    y = cy + math.sin(angle) * r

    speed = base_speed * (0.3 + 0.7 * random.random())
    vx = math.cos(angle) * speed + random.uniform(-spread, spread)
    vy = math.sin(angle) * speed + random.uniform(-spread, spread)

    # Star-like colors
    palette = [
        (255, 255, 255),
        (255, 240, 200),
        (200, 220, 255),
        (255, 210, 210),
        (210, 255, 230),
    ]
    color = random.choice(palette)
    radius = random.uniform(1.0, 2.8)
    life = random.uniform(3.5, 8.0)
    return Particle(x=x, y=y, vx=vx, vy=vy, radius=radius, color=color, life=life)


def compute_controls(hand_landmarks: list, w: int, h: int) -> dict:
    """
    hand_landmarks: list of normalized landmarks (x,y,z)
    Returns expansion [-1..+1], create [0..1], blackhole [0..1], swirl [-1..+1]
    """
    # Landmarks indices:
    # 0 wrist, 4 thumb tip, 8 index tip, 12 middle tip, 9 middle_mcp, 5 index_mcp
    def px(i: int) -> tuple[float, float]:
        lm = hand_landmarks[i]
        return lm.x * w, lm.y * h

    (wx, wy) = px(0)
    (mx, my) = px(9)
    (ixm, iym) = px(5)
    (tx, ty) = px(4)
    (ix, iy) = px(8)
    (midx, midy) = px(12)

    # Reference scale for normalization (palm-ish size)
    palm_ref = 0.5 * (math.hypot(mx - wx, my - wy) + math.hypot(ixm - wx, iym - wy))
    palm_ref = max(palm_ref, 30.0)

    d_ti = math.hypot(ix - tx, iy - ty)      # thumb-index
    d_tm = math.hypot(midx - tx, midy - ty)  # thumb-middle

    pinch_ti = d_ti / palm_ref
    pinch_tm = d_tm / palm_ref

    # Expansion control from thumb-index pinch
    # tune points: ~0.20 closed, ~1.00 open
    expansion = np.interp(pinch_ti, [0.20, 1.00], [-1.0, 1.0])
    expansion = float(clamp(expansion, -1.0, 1.0))

    # Blackhole when thumb-index is very close
    blackhole = (0.28 - pinch_ti) / 0.28
    blackhole = float(clamp(blackhole, 0.0, 1.0))

    # Creation burst when thumb-middle is close
    create = (0.30 - pinch_tm) / 0.30
    create = float(clamp(create, 0.0, 1.0))

    # Swirl direction from hand angle (wrist -> middle_mcp)
    ang = math.atan2(my - wy, mx - wx)  # radians
    swirl = math.sin(ang)  # -1..1, feels "alive" as you rotate hand

    return {
        "expansion": expansion,
        "blackhole": blackhole,
        "create": create,
        "swirl": float(clamp(swirl, -1.0, 1.0)),
        "pinch_ti": float(pinch_ti),
        "pinch_tm": float(pinch_tm),
    }


def main() -> None:
    # MediaPipe task model
    model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    if not os.path.exists(model_path):
        print("Error: hand_landmarker.task file nahi mili.")
        print("Download karke isi folder me rakho:")
        print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        sys.exit(1)

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: webcam open nahi ho raha.")
        sys.exit(1)

    # MediaPipe Tasks setup (works with your mp.tasks)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # Pygame setup
    pygame.init()
    W, H = 1100, 720
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Hand Universe: Expansion / Collapse / Creation / Destruction")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    cx, cy = W / 2, H / 2

    particles: list[Particle] = []
    max_particles = 3500

    # control smoothing
    expansion_s = 0.0
    create_s = 0.0
    blackhole_s = 0.0
    swirl_s = 0.0

    # spawn accumulator
    spawn_acc = 0.0

    SHOW_CAMERA = False  # True karoge to OpenCV camera window bhi dikhega

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        # Events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_q or e.key == pygame.K_ESCAPE:
                    running = False
                if e.key == pygame.K_c:
                    particles.clear()

        # Read camera frame
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        fh, fw, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Default controls if no hand
        controls = {
            "expansion": 0.0, "create": 0.0, "blackhole": 0.0, "swirl": 0.0,
            "pinch_ti": 0.0, "pinch_tm": 0.0
        }

        if result.hand_landmarks:
            controls = compute_controls(result.hand_landmarks[0], fw, fh)

        # Smooth controls
        smooth = 0.18
        expansion_s = lerp(expansion_s, controls["expansion"], smooth)
        create_s = lerp(create_s, controls["create"], smooth)
        blackhole_s = lerp(blackhole_s, controls["blackhole"], smooth)
        swirl_s = lerp(swirl_s, controls["swirl"], smooth)

        # Creation: spawn rate depends on create_s
        base_rate = 80.0     # particles/sec
        burst_rate = 900.0   # extra particles/sec at full pinch
        rate = base_rate + burst_rate * create_s

        spawn_acc += rate * dt
        spawn_n = int(spawn_acc)
        spawn_acc -= spawn_n

        if spawn_n > 0:
            spawn_n = min(spawn_n, 120)  # cap per frame
            for _ in range(spawn_n):
                if len(particles) >= max_particles:
                    break
                p = spawn_particle(cx, cy, base_speed=220.0, spread=55.0)
                particles.append(p)

        # Update particles
        alive: list[Particle] = []
        for p in particles:
            if p.update(dt, cx, cy, expansion_s, swirl_s, blackhole_s):
                alive.append(p)
        particles = alive

        # Draw background
        screen.fill((3, 2, 10))

        # Center glow (galactic core / black hole)
        core = (int(cx), int(cy))
        core_r = int(10 + 30 * blackhole_s + 18 * max(0.0, expansion_s))
        pygame.draw.circle(screen, (40, 20, 70), core, core_r)
        pygame.draw.circle(screen, (110, 80, 160), core, max(8, core_r // 2), width=2)

        # Draw particles
        for p in particles:
            p.draw(screen)

        # HUD text
        lines = [
            "Controls:",
            "Thumb-Index open  -> Expansion",
            "Thumb-Index pinch -> Collapse + Destruction (black hole)",
            "Thumb-Middle pinch-> Creation (new stars)",
            "Keys: Q/Esc quit | C clear",
            f"Particles: {len(particles)}/{max_particles}",
            f"expansion={expansion_s:+.2f}  create={create_s:.2f}  blackhole={blackhole_s:.2f}  swirl={swirl_s:+.2f}",
            f"pinchTI={controls['pinch_ti']:.2f}  pinchTM={controls['pinch_tm']:.2f}",
        ]
        y = 10
        for s in lines:
            surf = font.render(s, True, (200, 200, 220))
            screen.blit(surf, (10, y))
            y += 20

        pygame.display.flip()

        # Optional: show camera
        if SHOW_CAMERA:
            cv2.imshow("Camera (press Q in pygame window to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
