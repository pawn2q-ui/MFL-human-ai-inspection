"""
즉석 추론: 평가용 타일 캐시를 디스크에 만들지 않고, 매니페스트의 영상을 읽어 메모리에서 보정·타일링한 뒤
학습된 모델(여러 개 가능)로 예측하고 원본 5000x2400 좌표로 되돌려 저장한다.
출력 형식은 infer_tiles 와 같다 (evaluate / deferral 이 그대로 읽음).

  <out>/preds/<stem>.txt : "cls conf x1 y1 x2 y2" (원본 픽셀 좌표, conf >= conf_min 의 원시 예측)
  <out>/tile_stats.json  : 영상별 타일 통계
  <out>/DONE             : 완료 표시 (큐용)

사용 예 (시드 2개 모델을 한 번에 — 영상 읽기·타일링을 한 번만 하므로 빠름)
  python -m mfl_thesis.infer_images --manifest manifest.csv --cond C --split test,val ^
      --models runs/C_s0/weights/best.pt,runs/C_s1/weights/best.pt --outs preds/C_s0,preds/C_s1
  python -m mfl_thesis.infer_images --manifest manifest.csv --cond C --split test,val --wrap 0 ^
      --models runs/C_s0/weights/best.pt --outs preds/C_s0_nowrap
중단되면 같은 명령으로 다시 실행 → 이미 끝난 영상은 건너뜀.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PIN_MEMORY", "False")

import cv2
import numpy as np

from .config import IMG_W, IMG_H, A_CONF, NMS_IOU, DETECTOR_INPUT, A_WINDOW
from .geometry import resample_image, extract_tile, tile_origins, a_protocol_origins, merge_tile_predictions
from .labels import write_pred
from .make_tiles import default_s_star, median_train_diameter, factors_for
from .manifest import read_manifest


def plan(r, cond, s_star, D_med, T, overlap, wrap, scale=None):
    """영상 하나의 보정 배율·크기·타일 원점 (라벨 불필요). scale=(r_x, r_y) 를 주면 조건과 무관하게 그 배율 사용."""
    r_x, r_y = scale if scale else factors_for(cond, r["diameter_mm"], s_star, D_med)
    if cond == "A" and not scale:
        return 1.0, 1.0, IMG_W, IMG_H, A_WINDOW, False, a_protocol_origins("infer")
    W, H = max(1, int(round(IMG_W * r_x))), max(1, int(round(IMG_H * r_y)))
    return r_x, r_y, W, H, T, bool(wrap), tile_origins(W, H, T, overlap, wrap)


def load_tiles(r, meta, jpeg=92):
    """영상 읽기 → 보정 → 타일(640) 배열 목록. GPU 와 겹쳐 돌리기 위해 별도 스레드에서 실행.
    jpeg>0 이면 학습 타일 캐시(JPEG 92)와 같은 압축을 메모리에서 거쳐 학습·추론 입력 분포를 맞춘다."""
    r_x, r_y, W, H, T, wrap, origins = meta
    img = cv2.imread(r["image"], cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(r["image"])
    if r_x != 1.0 or r_y != 1.0:
        img = resample_image(img, r_x, r_y)
    interp = cv2.INTER_AREA if T > DETECTOR_INPUT else cv2.INTER_LINEAR
    tiles = []
    for x0, y0 in origins:
        t = cv2.resize(extract_tile(img, x0, y0, T, wrap), (DETECTOR_INPUT, DETECTOR_INPUT), interpolation=interp)
        if jpeg:
            ok, buf = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg)])
            if ok:
                t = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        tiles.append(t)
    return tiles


def ultralytics_array_predictors(model_paths, device=None, half=True, conf_min=0.01, batch=16, imgsz=DETECTOR_INPUT):
    """모델 경로들 -> [predict_fn(list[np.ndarray]) -> list[(N,6) cls,conf,x1,y1,x2,y2 (640 좌표)]]"""
    from ultralytics import YOLO
    fns = []
    for mp in model_paths:
        model = YOLO(str(mp))

        def predict_fn(arrays, model=model):
            out = []
            for i in range(0, len(arrays), batch):
                results = model.predict(arrays[i:i + batch], imgsz=imgsz, conf=conf_min, verbose=False,
                                        device=device, half=half)
                for res in results:
                    if res.boxes is None or len(res.boxes) == 0:
                        out.append(np.zeros((0, 6))); continue
                    b = res.boxes
                    out.append(np.concatenate([b.cls.cpu().numpy()[:, None], b.conf.cpu().numpy()[:, None],
                                               b.xyxy.cpu().numpy()], axis=1))
            return out
        fns.append(predict_fn)
    return fns


def run(manifest, cond, splits, predict_fns, outs, s_star=None, T=1280, overlap=0.458, wrap=True,
        conf_thr=A_CONF, nms_iou=NMS_IOU, limit=None, save_every=500, verbose=True, jpeg=92, pipes=None, scale=None):
    rows = read_manifest(manifest)
    sel = [r for r in rows if r["split"] in splits and (pipes is None or r["pipe_id"] in pipes)]
    if limit:
        sel = sel[:limit]
    s_star = s_star or default_s_star(rows)
    D_med = median_train_diameter(rows)
    outs = [Path(o) for o in outs]
    assert len(outs) == len(predict_fns)
    stats = []
    for o in outs:
        (o / "preds").mkdir(parents=True, exist_ok=True)
        f = o / "tile_stats.json"
        stats.append(json.loads(f.read_text(encoding="utf-8"))["images"] if f.exists() else {})

    def done(stem):
        return all(stem in st and (o / "preds" / f"{stem}.txt").exists() for st, o in zip(stats, outs))

    todo = [r for r in sel if not done(r["stem"])]
    if verbose:
        print(f"조건 {cond} | 영상 {len(sel)}장 중 {len(todo)}장 남음 | 모델 {len(outs)}개 | s*={s_star:.4f}, D_med={D_med}, T={T}, wrap={wrap}")

    def save_stats():
        for st, o in zip(stats, outs):
            (o / "tile_stats.json").write_text(json.dumps({"cond": cond, "images": st}, ensure_ascii=False))

    metas = {r["stem"]: plan(r, cond, s_star, D_med, T, overlap, wrap, scale) for r in todo}
    if verbose and scale:
        print(f"배율 고정: r_x={scale[0]:.3f}, r_y={scale[1]:.3f} (모든 영상)")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(load_tiles, todo[0], metas[todo[0]["stem"]], jpeg) if todo else None
        for i, r in enumerate(todo):
            tiles = fut.result()
            if i + 1 < len(todo):                       # 다음 영상 미리 읽기
                fut = ex.submit(load_tiles, todo[i + 1], metas[todo[i + 1]["stem"]], jpeg)
            r_x, r_y, W, H, T_eff, wrap_eff, origins = metas[r["stem"]]
            scale = T_eff / DETECTOR_INPUT
            for predict_fn, st, o in zip(predict_fns, stats, outs):
                preds640 = predict_fn(tiles)
                tile_preds, tstats = [], []
                for (x0, y0), p in zip(origins, preds640):
                    p = p.astype(float).copy()
                    if len(p):
                        p[:, 2:6] *= scale
                    tile_preds.append((x0, y0, p))
                    tstats.append({"x0": int(x0), "y0": int(y0),
                                   "max_conf": float(p[:, 1].max()) if len(p) else 0.0,
                                   "n_box_thr": int((p[:, 1] >= conf_thr).sum()) if len(p) else 0,
                                   "n_box_raw": int(len(p))})
                merged = merge_tile_predictions(tile_preds, W, H, r_x, r_y, wrap_eff, nms_iou)
                write_pred(o / "preds" / f"{r['stem']}.txt", merged)
                st[r["stem"]] = {"pipe": r["pipe_id"], "D": r["diameter_mm"], "split": r["split"],
                                 "n_tiles": len(origins), "tiles": tstats,
                                 "n_pred_thr": int((merged[:, 1] >= conf_thr).sum()) if len(merged) else 0}
            if verbose and (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(todo)} 영상, {el / 60:.1f}분 경과, 남은 예상 {(len(todo) - i - 1) * el / (i + 1) / 60:.0f}분")
            if (i + 1) % save_every == 0:
                save_stats()
    save_stats()
    for o in outs:
        (o / "DONE").write_text("ok")
    if verbose:
        print(f"추론 완료: {len(sel)} 영상 → {', '.join(str(o) for o in outs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cond", required=True, choices=["A", "B", "C"])
    ap.add_argument("--split", default="test,val")
    ap.add_argument("--models", required=True, help="쉼표 구분 모델 경로들")
    ap.add_argument("--outs", required=True, help="쉼표 구분 출력 폴더들 (모델 수와 같게)")
    ap.add_argument("--s-star", type=float, default=None)
    ap.add_argument("--T", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.458)
    ap.add_argument("--wrap", type=int, default=1)
    ap.add_argument("--conf-min", type=float, default=0.01)
    ap.add_argument("--conf-thr", type=float, default=A_CONF)
    ap.add_argument("--device", default=None)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jpeg", type=int, default=92, help="타일을 메모리에서 JPEG 압축/해제 (0=끔). 학습 캐시와 동일 조건")
    ap.add_argument("--pipes", default=None, help="쉼표 구분 배관 ID 로 제한 (예: A,B,C)")
    ap.add_argument("--scale", default=None, help="배율 고정 'r_x,r_y' (추론 배율 실험용, 조건 배율 대신 사용)")
    a = ap.parse_args()
    pipes = set(a.pipes.split(",")) if a.pipes else None
    scale = tuple(float(v) for v in a.scale.split(",")) if a.scale else None
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    outs = [o.strip() for o in a.outs.split(",") if o.strip()]
    fns = ultralytics_array_predictors(models, a.device, bool(a.half), a.conf_min, a.batch)
    run(a.manifest, a.cond, set(a.split.split(",")), fns, outs, a.s_star, a.T, a.overlap, bool(a.wrap),
        a.conf_thr, NMS_IOU, a.limit, jpeg=a.jpeg, pipes=pipes, scale=scale)


if __name__ == "__main__":
    main()
