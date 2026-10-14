"""
타일 캐시(infer 모드)에 학습된 모델을 적용하고, 예측을 원본 5000x2400 좌표로 되돌려 영상별 예측 파일로 저장.

출력
  <out>/preds/<stem>.txt : "cls conf x1 y1 x2 y2" (원본 픽셀 좌표, conf >= conf_min 의 '원시' 예측)
  <out>/tile_stats.json  : 영상별 타일 통계 (판단보류 점수용: 타일별 최대 신뢰도, 임계값 이상 박스 수)

사용 예 (지훈 PC, ultralytics 설치 후)
  python -m mfl_thesis.infer_tiles --tiles tiles/C_eval --model runs/C_s0/weights/best.pt --out preds/C_s0 --conf-min 0.01

테스트에서는 predict_fn 을 바꿔 끼워 모델 없이도 동작을 검증한다.
"""
import argparse
import os
os.environ.setdefault("PIN_MEMORY", "False")
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from .config import A_CONF, NMS_IOU, DETECTOR_INPUT
from .labels import write_pred
from .geometry import merge_tile_predictions


def ultralytics_predictor(model_path, device=None, batch=32, conf_min=0.01, imgsz=DETECTOR_INPUT):
    """ultralytics YOLO 모델 -> predict_fn(list[Path]) -> list[np.ndarray (N,6) cls,conf,x1,y1,x2,y2 (640 좌표)]"""
    from ultralytics import YOLO
    model = YOLO(model_path)

    def predict_fn(paths):
        out = []
        for i in range(0, len(paths), batch):
            chunk = [str(p) for p in paths[i:i + batch]]
            results = model.predict(chunk, imgsz=imgsz, conf=conf_min, verbose=False, device=device)
            for res in results:
                if res.boxes is None or len(res.boxes) == 0:
                    out.append(np.zeros((0, 6))); continue
                b = res.boxes
                arr = np.concatenate([b.cls.cpu().numpy()[:, None], b.conf.cpu().numpy()[:, None], b.xyxy.cpu().numpy()], axis=1)
                out.append(arr)
        return out
    return predict_fn


def run_inference(tiles_dir, predict_fn, out_dir, conf_thr=A_CONF, nms_iou=NMS_IOU, verbose=True):
    tiles_dir, out_dir = Path(tiles_dir), Path(out_dir)
    info = json.loads((tiles_dir / "tiles.json").read_text())
    T = info["T"]; wrap = info["wrap"]
    scale = T / info["detector_input"]          # 640 좌표 -> 타일 좌표
    by_img = defaultdict(list)
    for t in info["tiles"]:
        by_img[t["stem"]].append(t)
    paths = [tiles_dir / "images" / (t["tile"] + ".jpg") for t in info["tiles"]]
    preds640 = predict_fn(paths)
    pred_by_tile = {t["tile"]: p for t, p in zip(info["tiles"], preds640)}

    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    stats = {}
    for stem, tl in by_img.items():
        meta = info["images"][stem]
        tile_preds, tstats = [], []
        for t in tl:
            p = pred_by_tile[t["tile"]].astype(float).copy()
            if len(p):
                p[:, 2:6] *= scale
            tile_preds.append((t["x0"], t["y0"], p))
            tstats.append({"x0": t["x0"], "y0": t["y0"],
                           "max_conf": float(p[:, 1].max()) if len(p) else 0.0,
                           "n_box_thr": int((p[:, 1] >= conf_thr).sum()) if len(p) else 0,
                           "n_box_raw": int(len(p))})
        merged = merge_tile_predictions(tile_preds, meta["W"], meta["H"], meta["r_x"], meta["r_y"], wrap, nms_iou)
        write_pred(out_dir / "preds" / f"{stem}.txt", merged)
        stats[stem] = {"pipe": meta["pipe"], "D": meta["D"], "split": meta["split"], "n_tiles": len(tl), "tiles": tstats,
                       "n_pred_thr": int((merged[:, 1] >= conf_thr).sum()) if len(merged) else 0}
    (out_dir / "tile_stats.json").write_text(json.dumps({"cond": info["cond"], "images": stats}, ensure_ascii=False))
    if verbose:
        print(f"추론 완료: {len(by_img)} 영상 -> {out_dir}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf-min", type=float, default=0.01, help="저장할 원시 예측의 최소 신뢰도")
    ap.add_argument("--conf-thr", type=float, default=A_CONF, help="통계용 운영 임계값")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    run_inference(a.tiles, ultralytics_predictor(a.model, a.device, a.batch, a.conf_min), a.out, a.conf_thr)


if __name__ == "__main__":
    main()
