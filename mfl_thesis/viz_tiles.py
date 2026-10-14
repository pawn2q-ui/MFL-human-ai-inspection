"""
타일·라벨 시각 검증 (2주차 10/17 생명선용).
타일 캐시(make_tiles 출력)에서 양성 타일을 무작위로 골라, 왼쪽에는 640 타일 + 타일 라벨 박스,
오른쪽에는 원본 영상(축소) + 원본 라벨 박스 + 타일이 잘린 위치(빨간 테두리)를 나란히 그린다.
타일 라벨이 타일 위의 결함 무늬와 맞고, 빨간 테두리가 원본에서 같은 자리를 가리키면 기하 변환이 맞는 것이다.
원주 이음부(위/아래 경계)를 넘어가는 타일은 테두리가 위아래 두 조각으로 그려진다.

사용:
  python -m mfl_thesis.viz_tiles --tiles tiles/check_C --manifest manifest.csv --n 6 --out analysis/check_C.png
  (--seam 1 을 주면 이음부를 넘는 타일만 고른다)
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import IMG_W, IMG_H, CLASSES, DAMAGE
from .labels import read_yolo
from .manifest import read_manifest

COLORS = {"MTL": (214, 120, 42), "CRC": (52, 104, 235), "GWA": (122, 175, 27), "SWA": (0, 161, 237)}   # BGR
OTHER = (160, 160, 160)


def draw_boxes(img, boxes, scale=1.0, thick=2):
    for c, x1, y1, x2, y2 in boxes:
        name = CLASSES[int(c)]
        col = COLORS.get(name, OTHER)
        p1 = (int(x1 * scale), int(y1 * scale)); p2 = (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(img, p1, p2, col, thick)
        cv2.putText(img, name, (p1[0], max(12, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="make_tiles 출력 폴더 (images/, labels/, tiles.json)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seam", type=int, default=0, help="1: 이음부를 넘는 타일만")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tdir = Path(a.tiles)
    info = json.loads((tdir / "tiles.json").read_text(encoding="utf-8"))
    rows = {r["stem"]: r for r in read_manifest(a.manifest)}
    cand = [t for t in info["tiles"] if t["n_box"] > 0]
    if a.seam:
        cand = [t for t in cand if t["y0"] + info["images"][t["stem"]]["T"] > info["images"][t["stem"]]["H"]]
    if not cand:
        raise SystemExit("조건에 맞는 양성 타일이 없습니다")
    rng = random.Random(a.seed)
    picks = rng.sample(cand, min(a.n, len(cand)))

    fig, axes = plt.subplots(len(picks), 2, figsize=(11, 3.3 * len(picks)), dpi=110,
                             gridspec_kw={"width_ratios": [1, 2.1]})
    axes = np.atleast_2d(axes)
    for i, t in enumerate(picks):
        meta = info["images"][t["stem"]]
        r = rows[t["stem"]]
        # 왼쪽: 타일 + 타일 라벨
        tile = cv2.imread(str(tdir / "images" / (t["tile"] + ".jpg")), cv2.IMREAD_COLOR)
        tb = read_yolo(tdir / "labels" / (t["tile"] + ".txt"), tile.shape[1], tile.shape[0])
        draw_boxes(tile, tb, 1.0, 2)
        axes[i, 0].imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"{t['tile']}  (pipe {t['pipe']}, D={meta['D']:.0f} mm, boxes {t['n_box']})", fontsize=8)
        # 오른쪽: 원본(축소) + 원본 라벨 + 타일 위치
        img = cv2.imread(r["image"], cv2.IMREAD_COLOR)
        f = 1000 / IMG_W
        small = cv2.resize(img, (int(IMG_W * f), int(IMG_H * f)), interpolation=cv2.INTER_AREA)
        ob = read_yolo(r["label"], IMG_W, IMG_H) if r["label"] else np.zeros((0, 5))
        draw_boxes(small, ob, f, 1)
        T = meta["T"]; rx, ry = meta["r_x"], meta["r_y"]
        x0, y0 = t["x0"] / rx, t["y0"] / ry            # 원본 좌표
        w, h = T / rx, T / ry
        for (ya, yb) in ((y0, y0 + h), (y0 - IMG_H, y0 - IMG_H + h)):
            ca, cb = max(ya, 0), min(yb, IMG_H)
            if cb - ca > 1:
                cv2.rectangle(small, (int(x0 * f), int(ca * f)), (int((x0 + w) * f), int(cb * f)), (0, 0, 255), 2)
        axes[i, 1].imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"{t['stem']}  r_x={rx:.3f}, r_y={ry:.3f}, T={T}, wrap={meta['wrap']}", fontsize=8)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"cond {info['cond']} / {info['mode']} -- left: tile + tile labels | right: original + original labels, red box = tile location",
                 fontsize=9)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("->", a.out)


if __name__ == "__main__":
    main()
