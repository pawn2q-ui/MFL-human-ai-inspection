"""
관경 기반 공간 보정(4.1절)과 타일 구성(4.2절)의 핵심 기하 연산.

좌표계 요약
  원본  : 5000 x 2400 px (x: 축 방향 1 mm/px, y: 원주 방향 s_y(D) mm/px)
  보정  : 원본에 (r_x, r_y) 배율 적용. r_x = 1/s*, r_y = s_y(D)/s*  -> 두 축 모두 s* mm/px
  타일  : 보정 영상에서 T x T px 정사각 (물리 크기 T*s* mm). 세로는 원주 순환(wrap) 가능
  입력  : 타일을 640 x 640 으로 축소 (배율 640/T)
예측은 역순으로 되돌린다: 입력 -> 타일 -> 보정 -> 원본.
"""
import math
from typing import List, Tuple
import numpy as np
import cv2

from .config import IMG_W, IMG_H, AXIAL_MM_PER_PX, A_WINDOW, A_TRAIN_X0, A_INFER_STRIDE, s_y
from .labels import iou_matrix


# ----------------------------------------------------------------------------
# 배율
# ----------------------------------------------------------------------------
def scale_factors(diameter_mm: float, s_star: float) -> Tuple[float, float]:
    """조건 C: 배관별 배율 (r_x, r_y)."""
    return AXIAL_MM_PER_PX / s_star, s_y(diameter_mm) / s_star


def fixed_scale_factors(median_diameter_mm: float, s_star: float) -> Tuple[float, float]:
    """조건 B: 관경 중앙값 배관의 C 배율을 모든 배관에 동일 적용."""
    return scale_factors(median_diameter_mm, s_star)


def resample_image(img: np.ndarray, r_x: float, r_y: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_w, new_h = max(1, int(round(w * r_x))), max(1, int(round(h * r_y)))
    interp = cv2.INTER_AREA if (r_x < 1.0 and r_y < 1.0) else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def scale_boxes(boxes: np.ndarray, r_x: float, r_y: float) -> np.ndarray:
    """boxes: (N, k) 에서 열 [-4:] 가 x1,y1,x2,y2 라고 가정 (라벨 (N,5), 예측 (N,6) 모두 지원)."""
    b = boxes.astype(float).copy()
    if len(b):
        b[:, -4] *= r_x; b[:, -2] *= r_x
        b[:, -3] *= r_y; b[:, -1] *= r_y
    return b


# ----------------------------------------------------------------------------
# 타일 격자
# ----------------------------------------------------------------------------
def _uniform_origins(length: int, T: int, stride: int) -> List[int]:
    """[0, length-T] 구간을 균등 간격(<= stride)으로 덮는 시작점들. length <= T 이면 [0]."""
    if length <= T:
        return [0]
    n = int(math.ceil((length - T) / stride)) + 1
    return [int(round(i * (length - T) / (n - 1))) for i in range(n)]


def _wrap_origins(length: int, T: int, stride: int) -> List[int]:
    """원주(순환) 축: 둘레 length 를 T 타일로 균등하게 덮는 시작점들. length <= T 이면 [0]."""
    if length <= T:
        return [0]
    n = int(math.ceil(length / stride))
    return [int(round(i * length / n)) for i in range(n)]


def tile_origins(W: int, H: int, T: int, overlap: float = 0.458, wrap: bool = True) -> List[Tuple[int, int]]:
    """보정 영상(W x H)을 T x T 타일로 나누는 (x0, y0) 목록."""
    stride = max(1, int(round(T * (1.0 - overlap))))
    xs = _uniform_origins(W, T, stride)
    ys = _wrap_origins(H, T, stride) if wrap else _uniform_origins(H, T, stride)
    return [(x, y) for y in ys for x in xs]


def extract_tile(img: np.ndarray, x0: int, y0: int, T: int, wrap: bool = True) -> np.ndarray:
    """타일 추출. 세로는 wrap=True 면 둘레를 따라 순환, False 면 범위 밖을 0으로 채움. 가로 범위 밖은 0."""
    H, W = img.shape[:2]
    C = img.shape[2] if img.ndim == 3 else 1
    tile = np.zeros((T, T, C) if img.ndim == 3 else (T, T), dtype=img.dtype)
    cw = min(T, W - x0)
    if cw <= 0:
        return tile
    if wrap:
        rows = (np.arange(y0, y0 + T) % H)
        tile[:, :cw] = img[rows, x0:x0 + cw]
    else:
        ch = min(T, H - y0)
        if ch > 0:
            tile[:ch, :cw] = img[y0:y0 + ch, x0:x0 + cw]
    return tile


def boxes_for_tile(boxes: np.ndarray, x0: int, y0: int, T: int, H: int, wrap: bool = True,
                   min_frac: float = 0.5) -> np.ndarray:
    """보정 좌표 라벨 (N,5)[cls,x1,y1,x2,y2] -> 타일 좌표 라벨. 면적의 min_frac 이상이 타일 안에 있을 때만 유지(잘라서)."""
    out = []
    shifts = [0, H, -H] if wrap else [0]
    for c, x1, y1, x2, y2 in boxes:
        area = max((x2 - x1) * (y2 - y1), 1e-9)
        best = None
        for dy in shifts:
            bx1, by1, bx2, by2 = x1 - x0, y1 + dy - y0, x2 - x0, y2 + dy - y0
            ix1, iy1, ix2, iy2 = max(bx1, 0), max(by1, 0), min(bx2, T), min(by2, T)
            inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
            frac = inter / area
            if frac >= min_frac and (best is None or frac > best[0]):
                best = (frac, [c, ix1, iy1, ix2, iy2])
        if best is not None:
            out.append(best[1])
    return np.array(out, dtype=float).reshape(-1, 5)


def tile_preds_to_image(preds: np.ndarray, x0: int, y0: int, W: int, H: int, wrap: bool = True) -> np.ndarray:
    """타일 좌표 예측 (N,6)[cls,conf,x1,y1,x2,y2] -> 보정 영상 좌표. 원주 이음부를 넘는 박스는 둘로 나눔."""
    out = []
    for c, s, x1, y1, x2, y2 in preds:
        X1, X2 = np.clip([x1 + x0, x2 + x0], 0, W)
        Y1, Y2 = y1 + y0, y2 + y0
        if X2 - X1 <= 0:
            continue
        if not wrap:
            Y1, Y2 = np.clip([Y1, Y2], 0, H)
            if Y2 - Y1 > 0:
                out.append([c, s, X1, Y1, X2, Y2])
            continue
        if Y1 >= H:                      # 완전히 넘어감 -> 위로 되돌림
            Y1, Y2 = Y1 - H, Y2 - H
        if Y2 <= H:
            out.append([c, s, X1, Y1, X2, Y2])
        else:                            # 이음부에 걸침 -> 아래 조각 + 위 조각
            out.append([c, s, X1, Y1, X2, H])
            out.append([c, s, X1, 0, X2, Y2 - H])
    return np.array(out, dtype=float).reshape(-1, 6)


# ----------------------------------------------------------------------------
# NMS / 병합 / 역변환
# ----------------------------------------------------------------------------
def nms_classwise(preds: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    if len(preds) == 0:
        return preds.reshape(-1, 6)
    keep_all = []
    for c in np.unique(preds[:, 0]):
        p = preds[preds[:, 0] == c]
        order = np.argsort(-p[:, 1])
        p = p[order]
        keep = []
        suppressed = np.zeros(len(p), dtype=bool)
        ious = iou_matrix(p[:, 2:6], p[:, 2:6])
        for i in range(len(p)):
            if suppressed[i]:
                continue
            keep.append(i)
            suppressed |= ious[i] > iou_thr
            suppressed[i] = True
        keep_all.append(p[keep])
    return np.concatenate(keep_all, axis=0) if keep_all else preds.reshape(-1, 6)


def inverse_scale(preds: np.ndarray, r_x: float, r_y: float) -> np.ndarray:
    return scale_boxes(preds, 1.0 / r_x, 1.0 / r_y)


# ----------------------------------------------------------------------------
# 조건 A (공식 프로토콜): 2400x2400 창, 세로는 항상 둘레 전체
# ----------------------------------------------------------------------------
def a_protocol_origins(mode: str = "train") -> List[Tuple[int, int]]:
    if mode == "train":
        return [(x, 0) for x in A_TRAIN_X0]
    xs = list(range(0, IMG_W - A_WINDOW + 1, A_INFER_STRIDE))
    if xs[-1] != IMG_W - A_WINDOW:
        xs.append(IMG_W - A_WINDOW)
    return [(x, 0) for x in xs]


# ----------------------------------------------------------------------------
# 한 영상 전체 파이프라인 (조건 B/C 공용)
# ----------------------------------------------------------------------------
def tiles_from_image(img: np.ndarray, boxes: np.ndarray, r_x: float, r_y: float, T: int,
                     overlap: float = 0.458, wrap: bool = True, min_frac: float = 0.5):
    """원본 영상+라벨 -> 보정 -> 타일 목록. 반환: (resampled_shape, [(x0,y0,tile_img,tile_boxes), ...])"""
    rs = resample_image(img, r_x, r_y)
    H, W = rs.shape[:2]
    rb = scale_boxes(boxes, r_x, r_y)
    tiles = []
    for x0, y0 in tile_origins(W, H, T, overlap, wrap):
        tiles.append((x0, y0, extract_tile(rs, x0, y0, T, wrap), boxes_for_tile(rb, x0, y0, T, H, wrap, min_frac)))
    return (W, H), tiles


def merge_tile_predictions(tile_preds: List[Tuple[int, int, np.ndarray]], W: int, H: int,
                           r_x: float, r_y: float, wrap: bool = True, iou_thr: float = 0.5) -> np.ndarray:
    """[(x0,y0,preds_in_tile_coords(N,6)), ...] -> 원본 좌표 예측 (NMS 병합 후)."""
    all_p = [tile_preds_to_image(p, x0, y0, W, H, wrap) for x0, y0, p in tile_preds if len(p)]
    if not all_p:
        return np.zeros((0, 6))
    merged = nms_classwise(np.concatenate(all_p, axis=0), iou_thr)
    return inverse_scale(merged, r_x, r_y)
