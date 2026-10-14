"""
YOLO 라벨 입출력과 좌표 변환 유틸.
YOLO 형식: 한 줄에 "cls cx cy w h" (0~1 정규화). 내부 계산은 픽셀 xyxy 로 한다.
"""
from pathlib import Path
import numpy as np


def read_yolo(path, img_w, img_h) -> np.ndarray:
    """-> (N,5) float array: [cls, x1, y1, x2, y2] (픽셀). 없으면 (0,5)."""
    path = Path(path)
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            c = int(float(t[0])); cx, cy, w, h = map(float, t[1:5])
            rows.append([c, (cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h])
    return np.array(rows, dtype=float).reshape(-1, 5)


def write_yolo(path, boxes_xyxy, img_w, img_h):
    """boxes: (N,5) [cls,x1,y1,x2,y2] 픽셀 -> YOLO 정규화 저장."""
    lines = []
    for c, x1, y1, x2, y2 in boxes_xyxy:
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        cx, cy = x1 + w / 2, y1 + h / 2
        lines.append(f"{int(c)} {cx / img_w:.6f} {cy / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}")
    Path(path).write_text("\n".join(lines))


def read_pred(path) -> np.ndarray:
    """예측 파일: 한 줄에 "cls conf x1 y1 x2 y2" (픽셀, 원본 좌표) -> (N,6)."""
    path = Path(path)
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            t = line.split()
            if len(t) < 6:
                continue
            rows.append([int(float(t[0]))] + [float(v) for v in t[1:6]])
    return np.array(rows, dtype=float).reshape(-1, 6)


def write_pred(path, preds):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{int(c)} {s:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}" for c, s, x1, y1, x2, y2 in preds]
    Path(path).write_text("\n".join(lines))


def iou_matrix(a, b) -> np.ndarray:
    """a:(N,4) b:(M,4) xyxy -> (N,M) IoU"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)
