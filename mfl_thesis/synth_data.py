"""
가짜(합성) 데이터 생성과 모형 예측기 — 실제 데이터 없이 파이프라인 전체를 검증하기 위한 도구.

- 배관 5개(457~813 mm), 영상 5000x2400 의사색 3채널, 결함은 실제 크기(mm)로 생성한 뒤 픽셀로 환산해 그림
- 일부 결함은 원주 이음부(위/아래 경계)에 걸치게 두고, 라벨은 이음부에서 둘로 나눔(실데이터의 라벨 관행 가정)
- mock_predictor: 타일 라벨을 보고 '픽셀 크기가 작을수록 잘 놓치는' 탐지기를 흉내 냄 (관경 편향 재현)
"""
import csv
import random
from pathlib import Path
import numpy as np
import cv2

from .config import IMG_W, IMG_H, CLASSES, DAMAGE, s_y
from .labels import write_yolo, read_yolo

PIPES = {"A": 457.0, "B": 508.0, "C": 610.0, "D": 711.0, "E": 813.0}


def _background(rng):
    base = rng.normal(128, 6, (IMG_H // 8, IMG_W // 8)).astype(np.float32)
    base = cv2.resize(base, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)
    img = np.stack([base * 0.9, base, base * 1.1], axis=2)
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_blob(img, cx, cy, ax_px, circ_px, amp):
    """타원 블롭을 (원주 방향 순환을 고려해) 그린다."""
    for dy in (0, IMG_H, -IMG_H):
        c = (int(round(cx)), int(round(cy + dy)))
        if -circ_px <= c[1] <= IMG_H + circ_px:
            overlay = img.copy()
            cv2.ellipse(overlay, c, (max(1, int(ax_px / 2)), max(1, int(circ_px / 2))), 0, 0, 360, (int(amp), int(amp * 0.6), 255 - int(amp * 0.3)), -1)
            img[:] = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)


def generate(out_dir, n_per_pipe=8, empty_frac=0.3, seed=0, pipes=PIPES):
    out = Path(out_dir); (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed); rnd = random.Random(seed)
    with open(out / "pipes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pipe_id", "diameter_mm"]); [w.writerow([k, v]) for k, v in pipes.items()]
    for pid, D in pipes.items():
        sy = s_y(D)
        for i in range(n_per_pipe):
            img = _background(rng)
            boxes = []
            if rnd.random() >= empty_frac:
                for _ in range(rnd.randint(1, 4)):
                    cls = rnd.choice(DAMAGE) if rnd.random() < 0.85 else rnd.choice([c for c in CLASSES if c not in DAMAGE])
                    L_ax, L_circ = rnd.uniform(15, 150), rnd.uniform(10, 120)          # mm
                    ax_px, circ_px = L_ax, L_circ / sy                                 # px
                    cx = rnd.uniform(ax_px, IMG_W - ax_px)
                    cy = rnd.uniform(-circ_px * 0.4, IMG_H + circ_px * 0.4) if rnd.random() < 0.15 else rnd.uniform(circ_px, IMG_H - circ_px)
                    _draw_blob(img, cx, cy, ax_px, circ_px, rnd.uniform(170, 240))
                    x1, x2 = cx - ax_px / 2, cx + ax_px / 2
                    y1, y2 = cy - circ_px / 2, cy + circ_px / 2
                    # 이음부에서 둘로 나누기
                    for (a, b) in ((y1, y2), (y1 + IMG_H, y2 + IMG_H), (y1 - IMG_H, y2 - IMG_H)):
                        ca, cb = max(a, 0), min(b, IMG_H)
                        if cb - ca > 1:
                            boxes.append([CLASSES.index(cls), x1, ca, x2, cb])
            stem = f"{pid}_{i:05d}"
            cv2.imwrite(str(out / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            write_yolo(out / "labels" / f"{stem}.txt", np.array(boxes).reshape(-1, 5), IMG_W, IMG_H)
    return out


def mock_predictor(labels_dir, T, detector_input=640, seed=0, miss_scale_px=9.0, fp_rate=0.3, near_miss_p=0.6, perfect=False):
    """타일 라벨을 읽어 '탐지기 출력'을 흉내 낸다 (640 좌표). perfect=True 면 라벨을 그대로 반환(기하 왕복 검증용)."""
    rng = np.random.default_rng(seed)
    labels_dir = Path(labels_dir)
    sc = detector_input / T

    def predict_fn(paths):
        out = []
        for p in paths:
            gt = read_yolo(labels_dir / (Path(p).stem + ".txt"), T, T)   # 타일 좌표(px)
            preds = []
            for c, x1, y1, x2, y2 in gt:
                b = np.array([x1, y1, x2, y2]) * sc
                if perfect:
                    preds.append([c, 0.99, *b]); continue
                h640 = b[3] - b[1]
                p_det = 1 / (1 + np.exp(-(h640 - 14) / miss_scale_px))     # 픽셀 높이가 작을수록 놓침
                jit = rng.normal(0, 0.04, 4) * np.array([b[2] - b[0], b[3] - b[1]] * 2)
                # 같은 결함은 겹치는 타일들에서 일관되게 놓치도록, 박스 크기에서 유도한 난수를 사용
                key = (int(c) * 7919 + int(round((b[2] - b[0]) * 10)) * 131 + int(round(h640 * 10)) + seed * 17) % 10007
                u = np.random.default_rng(key).random()
                if u < p_det:
                    preds.append([c, float(np.clip(rng.uniform(0.3, 1.0) * (0.6 + 0.4 * p_det), 0.26, 0.99)), *(b + jit)])
                elif rng.random() < near_miss_p:
                    preds.append([c, float(rng.uniform(0.05, 0.24)), *(b + jit)])  # 임계값 아래 '근접 미탐'
            if not perfect:
                for _ in range(rng.poisson(fp_rate)):
                    x, y = rng.uniform(0, detector_input - 30, 2); w, h = rng.uniform(8, 40, 2)
                    preds.append([rng.choice(len(CLASSES)), float(rng.uniform(0.02, 0.6)), x, y, x + w, y + h])
            out.append(np.array(preds, dtype=float).reshape(-1, 6))
        return out
    return predict_fn
