"""
조건별 타일 캐시 생성 (4.2절). 타일은 640x640 JPG + YOLO 라벨로 저장하고, 원본 좌표로 되돌리는 데
필요한 정보를 tiles.json 에 기록한다.

조건
  A : 공식 프로토콜. 2400x2400 창(세로 = 둘레 전체), 학습 2개 / 추론 3개(간격 1300), 보정 없음
  B : 모든 배관 동일 배율(관경 중앙값 배관의 C 배율) + T px 정사각 타일 (+ wrap)
  C : 배관별 배율 r_x=1/s*, r_y=s_y(D)/s* + T px 정사각 타일 (+ wrap)

모드
  train : 결함(양성) 타일 전부 + 빈 타일을 pos:neg = 9:1 이 되도록 샘플링(공식 프로토콜과 동일)
  infer : 모든 타일 (평가·추론용), 라벨이 있으면 함께 기록

사용 예
  python -m mfl_thesis.make_tiles --manifest manifest.csv --cond C --split train --mode train \
      --T 1280 --overlap 0.458 --wrap 1 --out tiles/C_train
  python -m mfl_thesis.make_tiles --manifest manifest.csv --cond C --split test,val --mode infer --out tiles/C_eval
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import cv2
import yaml

from .config import IMG_W, IMG_H, CLASSES, DETECTOR_INPUT, A_WINDOW, s_y
from .labels import read_yolo, write_yolo
from .geometry import (scale_factors, fixed_scale_factors, resample_image, scale_boxes, tile_origins,
                       extract_tile, boxes_for_tile, a_protocol_origins)
from .manifest import read_manifest


def median_train_diameter(rows):
    ds = sorted({r["diameter_mm"] for r in rows if r["split"] == "train"})
    if not ds:
        raise ValueError("학습 배관이 없어 관경 중앙값을 정할 수 없습니다")
    return ds[len(ds) // 2]


def default_s_star(rows):
    return min(s_y(r["diameter_mm"]) for r in rows if r["split"] == "train")


def factors_for(cond, D, s_star, D_med):
    if cond == "A":
        return 1.0, 1.0
    if cond == "B":
        return fixed_scale_factors(D_med, s_star)
    return scale_factors(D, s_star)


def plan_image(r, cond, mode, s_star, D_med, T, overlap, wrap, min_frac):
    """영상을 읽지 않고(라벨만) 타일 격자와 타일별 라벨을 계산."""
    boxes = read_yolo(r["label"], IMG_W, IMG_H) if r["label"] else np.zeros((0, 5))
    r_x, r_y = factors_for(cond, r["diameter_mm"], s_star, D_med)
    if cond == "A":
        W, H = IMG_W, IMG_H
        origins = a_protocol_origins("train" if mode == "train" else "infer")
        Tn, wrap_eff = A_WINDOW, False
        rb = boxes
    else:
        W, H = max(1, int(round(IMG_W * r_x))), max(1, int(round(IMG_H * r_y)))
        origins = tile_origins(W, H, T, overlap, wrap)
        Tn, wrap_eff = T, wrap
        rb = scale_boxes(boxes, r_x, r_y)
    tiles = []
    for k, (x0, y0) in enumerate(origins):
        tb = boxes_for_tile(rb, x0, y0, Tn, H, wrap_eff, min_frac)
        tiles.append({"tile": f"{r['stem']}__t{k:03d}", "stem": r["stem"], "pipe": r["pipe_id"],
                      "x0": int(x0), "y0": int(y0), "n_box": int(len(tb)), "_boxes": tb})
    meta = {"W": int(W), "H": int(H), "r_x": float(r_x), "r_y": float(r_y), "T": int(Tn), "wrap": bool(wrap_eff),
            "pipe": r["pipe_id"], "D": r["diameter_mm"], "split": r["split"]}
    return tiles, meta


def render_tiles(r, meta, tiles_to_save, out_img, out_lbl):
    img = cv2.imread(r["image"], cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(r["image"])
    if meta["r_x"] != 1.0 or meta["r_y"] != 1.0:
        img = resample_image(img, meta["r_x"], meta["r_y"])
    T = meta["T"]
    interp = cv2.INTER_AREA if T > DETECTOR_INPUT else cv2.INTER_LINEAR
    for t in tiles_to_save:
        tile = extract_tile(img, t["x0"], t["y0"], T, meta["wrap"])
        small = cv2.resize(tile, (DETECTOR_INPUT, DETECTOR_INPUT), interpolation=interp)
        cv2.imwrite(str(out_img / (t["tile"] + ".jpg")), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
        write_yolo(out_lbl / (t["tile"] + ".txt"), t["_boxes"], T, T)   # 타일 좌표 정규화 -> 640 축소와 무관


def build(manifest, cond, splits, mode, out, s_star=None, T=1280, overlap=0.458, wrap=True, min_frac=0.5,
          neg_ratio=0.1, seed=0, limit=None, verbose=True, sample=None):
    rows = read_manifest(manifest)
    sel = [r for r in rows if r["split"] in splits]
    if sample:                                   # 무작위 N장 (시각 검증용; 양성 영상 우선)
        rs = random.Random(seed)
        pos = [r for r in sel if r["label"] and Path(r["label"]).stat().st_size > 0]
        sel = rs.sample(pos, min(sample, len(pos))) if pos else rs.sample(sel, min(sample, len(sel)))
    if limit:
        sel = sel[:limit]
    s_star = s_star or default_s_star(rows)
    D_med = median_train_diameter(rows)
    out = Path(out); out_img = out / "images"; out_lbl = out / "labels"
    out_img.mkdir(parents=True, exist_ok=True); out_lbl.mkdir(parents=True, exist_ok=True)

    # 1) 계획 (라벨만 읽음): 양성/빈 타일 수 집계
    plans = [plan_image(r, cond, mode, s_star, D_med, T, overlap, wrap, min_frac) for r in sel]
    n_pos = sum(t["n_box"] > 0 for tiles, _ in plans for t in tiles)
    n_emp = sum(t["n_box"] == 0 for tiles, _ in plans for t in tiles)
    if mode == "train":
        target_neg = int(round(n_pos * neg_ratio / max(1e-9, 1 - neg_ratio)))
        q = min(1.0, target_neg / max(1, n_emp))
    else:
        q = 1.0
    rng = random.Random(seed)

    # 2) 렌더링
    tiles_meta, images_meta = [], {}
    kept_pos = kept_neg = 0
    for i, (r, (tiles, meta)) in enumerate(zip(sel, plans)):
        keep = [t for t in tiles if t["n_box"] > 0 or rng.random() < q]
        if keep:
            render_tiles(r, meta, keep, out_img, out_lbl)
        images_meta[r["stem"]] = meta
        for t in keep:
            tiles_meta.append({k: v for k, v in t.items() if not k.startswith("_")})
            kept_pos += t["n_box"] > 0; kept_neg += t["n_box"] == 0
        if verbose and (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(sel)} 영상, 타일 {len(tiles_meta)}개 (pos {kept_pos}, neg {kept_neg})")

    T_eff = A_WINDOW if cond == "A" else T          # 조건 A 는 창 크기 2400 고정
    info = {"cond": cond, "mode": mode, "split": sorted(splits), "s_star": s_star, "D_med": D_med, "T": T_eff,
            "overlap": overlap, "wrap": bool(wrap), "min_frac": min_frac, "detector_input": DETECTOR_INPUT,
            "n_tiles": len(tiles_meta), "n_pos": int(kept_pos), "n_neg": int(kept_neg),
            "images": images_meta, "tiles": tiles_meta}
    (out / "tiles.json").write_text(json.dumps(info, ensure_ascii=False))
    yaml.safe_dump({"path": str(out.resolve()), "train": "images", "val": "images",
                    "names": {i: c for i, c in enumerate(CLASSES)}}, open(out / "data.yaml", "w"), allow_unicode=True)
    if verbose:
        print(f"완료: {out} | 타일 {len(tiles_meta)} (pos {kept_pos}, neg {kept_neg}) | s*={s_star:.4f} mm/px, D_med={D_med}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cond", required=True, choices=["A", "B", "C"])
    ap.add_argument("--split", required=True, help="train / val / test / adapt (쉼표로 여러 개)")
    ap.add_argument("--mode", default="train", choices=["train", "infer"])
    ap.add_argument("--s-star", type=float, default=None, help="공통 해상도 mm/px (기본: 학습 배관 최소 s_y)")
    ap.add_argument("--T", type=int, default=1280, help="타일 한 변 px (보정 좌표)")
    ap.add_argument("--overlap", type=float, default=0.458)
    ap.add_argument("--wrap", type=int, default=1)
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--neg-ratio", type=float, default=0.1, help="train 모드: 빈 타일 비율 (0.1 = 9:1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="디버그용: 영상 수 제한(앞에서부터)")
    ap.add_argument("--sample", type=int, default=None, help="디버그용: 양성 영상 무작위 N장")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.manifest, a.cond, set(a.split.split(",")), a.mode, a.out, a.s_star, a.T, a.overlap, bool(a.wrap),
          a.min_frac, a.neg_ratio, a.seed, a.limit, sample=a.sample)


if __name__ == "__main__":
    main()
