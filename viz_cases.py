"""
실패 사례 그림 (5장·포스터용): 미탐이 많은 영상과 오탐이 많은 영상을 골라 정답(초록)·예측(주황) 박스를 겹쳐 그린다.
  python -m mfl_thesis.viz_cases --manifest manifest.csv --split test --preds preds/A_s1/preds --pipes A,B,C ^
      --n-miss 3 --n-fp 3 --out analysis/cases_A_s1
영상 전체(가로 5 m)를 폭 2,000 px 로 줄인 그림과, 정답 박스 주변을 잘라 원본 해상도로 보여 주는 확대 그림을 저장한다.
"""
import argparse
from pathlib import Path

import numpy as np
import cv2

from .config import IMG_W, IMG_H, A_CONF, DAMAGE_IDS, CLASSES
from .labels import read_yolo, read_pred
from .manifest import read_manifest
from .evaluate import match_image

GREEN, ORANGE, RED = (60, 180, 75), (40, 120, 235), (40, 40, 230)   # BGR


def draw(img, boxes, color, labels=None, thick=2):
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        if labels is not None:
            cv2.putText(img, labels[i], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def render_case(r, gt, pr, out_dir, tag, thr=A_CONF, crop_pad=120):
    img = cv2.imread(r["image"], cv2.IMREAD_COLOR)
    if img is None:
        return
    hit, _, tp = match_image(gt, pr)
    gt_lab = [f"{CLASSES[int(c)]}{'' if h else ' MISS'}" for c, h in zip(gt[:, 0], hit)]
    pr_lab = [f"{CLASSES[int(p[0])]} {p[1]:.2f}{'' if t else ' FP'}" for p, t in zip(pr, tp)]
    full = img.copy()
    draw(full, gt[:, 1:5], GREEN, gt_lab, 3)
    draw(full, pr[:, 2:6], ORANGE, pr_lab, 2)
    scale = 2000 / IMG_W
    small = cv2.resize(full, (2000, int(IMG_H * scale)), interpolation=cv2.INTER_AREA)
    cv2.putText(small, f"{r['stem']} ({r['pipe_id']}, D={r['diameter_mm']:.0f} mm) green=GT orange=pred  miss={int((~hit).sum())}/{len(gt)} fp={int((~tp).sum())}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{tag}_{r['stem']}_full.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    # 확대: 미탐 정답 또는 오탐 예측 주변 (최대 4개)
    crops = []
    for b, h in zip(gt[:, 1:5], hit):
        if not h:
            crops.append(b)
    for p, t in zip(pr[:, 2:6], tp):
        if not t and len(crops) < 4:
            crops.append(p)
    for k, b in enumerate(crops[:4]):
        x1 = max(0, int(b[0]) - crop_pad); y1 = max(0, int(b[1]) - crop_pad)
        x2 = min(IMG_W, int(b[2]) + crop_pad); y2 = min(IMG_H, int(b[3]) + crop_pad)
        crop = full[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        h, w = crop.shape[:2]
        f = min(3.0, 600 / max(h, w))
        crop = cv2.resize(crop, (int(w * f), int(h * f)), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / f"{tag}_{r['stem']}_crop{k}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--pipes", default=None)
    ap.add_argument("--n-miss", type=int, default=3)
    ap.add_argument("--n-fp", type=int, default=3)
    ap.add_argument("--thr", type=float, default=A_CONF)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pipes = set(a.pipes.split(",")) if a.pipes else None
    rows = [r for r in read_manifest(a.manifest) if r["split"] in set(a.split.split(",")) and (pipes is None or r["pipe_id"] in pipes)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stats = []
    cache = {}
    for r in rows:
        gt = read_yolo(r["label"], IMG_W, IMG_H) if r["label"] else np.zeros((0, 5))
        pr = read_pred(Path(a.preds) / f"{r['stem']}.txt")
        pr = pr[pr[:, 1] >= a.thr] if len(pr) else pr
        gt_d = gt[np.isin(gt[:, 0], DAMAGE_IDS)] if len(gt) else gt
        hit, _, tp = match_image(gt, pr)
        n_miss = int((~hit[np.isin(gt[:, 0], DAMAGE_IDS)]).sum()) if len(gt) else 0
        n_fp = int((~tp).sum())
        stats.append((r["stem"], n_miss, n_fp)); cache[r["stem"]] = (r, gt, pr)
    by_miss = sorted(stats, key=lambda x: -x[1])[:a.n_miss]
    by_fp = sorted(stats, key=lambda x: -x[2])[:a.n_fp]
    for s in by_miss:
        r, gt, pr = cache[s[0]]; render_case(r, gt, pr, out, "miss", a.thr)
    for s in by_fp:
        r, gt, pr = cache[s[0]]; render_case(r, gt, pr, out, "fp", a.thr)
    (out / "DONE").write_text("ok")
    print("미탐 사례:", [(s[0], s[1]) for s in by_miss]); print("오탐 사례:", [(s[0], s[2]) for s in by_fp]); print("저장:", out.resolve())


if __name__ == "__main__":
    main()
