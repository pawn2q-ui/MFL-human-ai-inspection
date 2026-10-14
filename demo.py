"""
시연 프로그램 (경진대회 '시스템 적용 가능성'): MFL 영상 한 장 + 공칭 외경 → 결함 박스(mm 크기) + 검토 권고.

기본은 논문의 최종 권고인 공식 프로토콜(조건 A: 2,400 px 창 3개 → 640, NMS 0.5)이다.
  python -m mfl_thesis.demo --image Test_A_0000468.png --diameter 457 --model runs/A_s1/weights/best.pt --out demo_out
보정·타일링 조건(C 계열)으로 돌리려면:
  python -m mfl_thesis.demo --image ... --diameter 711 --model runs/C_s0/weights/best.pt --cond C --s-star 0.7985 --T 1280 --out demo_out

출력: demo_out/<stem>_pred.jpg (박스·신뢰도·mm 크기, 검토 권고 배너), demo_out/<stem>.json (박스 목록, 점수, 권고)
검토 권고(5.5절): 운영 임계값 이상 검출 수(n_pred)와 임계값 미만 약한 검출의 최대 신뢰도(sub_thr_max)를
시험 배관 13,740장의 분포(review_reference.json)와 비교하여 상위 백분위를 구하고, 어느 하나라도 상위 --review-top % 안이면 '검사원 검토 권고'.
여러 장을 한 번에 돌리려면 --image 에 폴더를 주면 된다(폴더 안 png/jpg 전부).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import cv2

from .config import IMG_W, IMG_H, CLASSES, DAMAGE_IDS, A_CONF, NMS_IOU, DETECTOR_INPUT, A_WINDOW, s_y
from .geometry import scale_factors, resample_image, tile_origins, extract_tile, merge_tile_predictions, a_protocol_origins
from .deferral import extrapolation_term

REF = Path(__file__).parent / "review_reference.json"


def percentile_of(value, quantiles):
    """값이 기준 분포(101개 분위수)에서 몇 번째 백분위인지 (0–100, 클수록 상위)."""
    q = np.asarray(quantiles, dtype=float)
    return float((q <= value).mean() * 100.0)


def run_one(model, img_path, diameter, cond, s_star, T, overlap, wrap, conf, lam, s_range, review_top, ref, out_dir, jpeg=92):
    t0 = time.time()
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)
    if cond == "A":
        r_x = r_y = 1.0; rs = img; W, H = IMG_W, IMG_H; T_eff = A_WINDOW; use_wrap = False
        origins = a_protocol_origins("infer")
    else:
        r_x, r_y = scale_factors(diameter, s_star)
        rs = resample_image(img, r_x, r_y); H, W = rs.shape[:2]; T_eff = T; use_wrap = bool(wrap)
        origins = tile_origins(W, H, T_eff, overlap, use_wrap)
    interp = cv2.INTER_AREA if T_eff > DETECTOR_INPUT else cv2.INTER_LINEAR
    tiles = []
    for x0, y0 in origins:
        t = cv2.resize(extract_tile(rs, x0, y0, T_eff, use_wrap), (DETECTOR_INPUT, DETECTOR_INPUT), interpolation=interp)
        if jpeg:   # 학습 캐시(JPEG 92)와 같은 압축을 거쳐 분포를 맞춘다
            ok, buf = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg)])
            if ok:
                t = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        tiles.append(t)
    results = model.predict(tiles, imgsz=DETECTOR_INPUT, conf=0.01, verbose=False)
    sc = T_eff / DETECTOR_INPUT
    tile_preds = []
    for (x0, y0), res in zip(origins, results):
        if res.boxes is None or len(res.boxes) == 0:
            tile_preds.append((x0, y0, np.zeros((0, 6)))); continue
        b = res.boxes
        arr = np.concatenate([b.cls.cpu().numpy()[:, None], b.conf.cpu().numpy()[:, None], b.xyxy.cpu().numpy() * sc], axis=1)
        tile_preds.append((x0, y0, arr))
    preds = merge_tile_predictions(tile_preds, W, H, r_x, r_y, use_wrap, NMS_IOU)
    sy = s_y(diameter)
    op = preds[preds[:, 1] >= conf] if len(preds) else np.zeros((0, 6))
    sub = preds[preds[:, 1] < conf][:, 1] if len(preds) else np.zeros(0)
    n_pred = int(len(op)); sub_max = float(sub.max()) if len(sub) else 0.0
    e_p = 0.0
    if s_range:
        smin, smax = map(float, s_range.split(","))
        e_p = extrapolation_term(diameter, smin, smax)
    # 검토 권고: 기준 분포 대비 상위 백분위
    key = cond if cond in ref else "A"
    pct_n = percentile_of(n_pred, ref[key]["n_pred"]); pct_s = percentile_of(sub_max, ref[key]["sub"])
    top_n, top_s = 100.0 - pct_n, 100.0 - pct_s          # 상위 몇 %인가 (작을수록 상위)
    review = (top_n < review_top) or (top_s < review_top) or (lam * e_p >= 0.5)
    reason = []
    if top_n < review_top: reason.append(f"검출 수 {n_pred}개 (상위 {top_n:.0f}%)")
    if top_s < review_top: reason.append(f"약한 검출 신뢰도 {sub_max:.2f} (상위 {top_s:.0f}%)")
    if lam * e_p >= 0.5: reason.append(f"학습 범위 밖 관경 (외삽 {e_p:.2f})")
    # 시각화
    vis = img.copy()
    for c, s, x1, y1, x2, y2 in op:
        col = (0, 0, 255) if int(c) in DAMAGE_IDS else (0, 200, 0)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
        cv2.putText(vis, f"{CLASSES[int(c)]} {s:.2f} | {(x2 - x1):.0f}x{(y2 - y1) * sy:.0f}mm", (int(x1), max(28, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
    n_dmg = int(np.isin(op[:, 0], DAMAGE_IDS).sum()) if len(op) else 0
    banner = (f"D={diameter:.0f}mm  s_y={sy:.3f}mm/px  cond={cond}  damage boxes={n_dmg}  n_pred={n_pred}  sub_max={sub_max:.2f}  "
              + ("REVIEW RECOMMENDED: " + "; ".join(reason) if review else "AUTO (no review)"))
    cv2.rectangle(vis, (0, 0), (IMG_W, 70), (0, 0, 0), -1)
    cv2.putText(vis, banner, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 80, 255) if review else (255, 255, 255), 3)
    out_dir.mkdir(parents=True, exist_ok=True); stem = Path(img_path).stem
    cv2.imwrite(str(out_dir / f"{stem}_pred.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    info = {"image": str(img_path), "diameter_mm": diameter, "s_y_mm_per_px": sy, "cond": cond, "n_windows": len(origins),
            "elapsed_s": round(time.time() - t0, 2), "n_pred": n_pred, "sub_thr_max": sub_max, "top_pct_n_pred": round(top_n, 1),
            "top_pct_sub": round(top_s, 1), "extrapolation": e_p, "review_recommended": bool(review), "reason": reason,
            "detections": [{"cls": CLASSES[int(c)], "conf": round(float(s), 3), "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                            "axial_mm": float(x2 - x1), "circ_mm": float((y2 - y1) * sy)} for c, s, x1, y1, x2, y2 in op]}
    (out_dir / f"{stem}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="영상 파일 또는 폴더")
    ap.add_argument("--diameter", type=float, required=True, help="공칭 외경 (mm)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cond", default="A", choices=["A", "B", "C", "Cp"], help="A = 공식 프로토콜(기본), C 계열 = 보정·타일링")
    ap.add_argument("--s-star", type=float, default=None, help="C 계열 공통 해상도 (예: 0.7985, C′는 0.5982)")
    ap.add_argument("--T", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.458); ap.add_argument("--wrap", type=int, default=1)
    ap.add_argument("--s-range", default="0.7985,1.0640", help="학습 배관 s_y 범위 'min,max' (외삽 항)")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.0, help="관경 외삽 항 가중치 (논문 기본 0; 1이면 학습 범위 밖 관경을 검토 권고)")
    ap.add_argument("--conf", type=float, default=A_CONF)
    ap.add_argument("--review-top", type=float, default=20.0, help="검출 수 또는 약한 검출이 시험 분포 상위 몇 %% 안이면 검토 권고")
    ap.add_argument("--out", default="demo_out")
    a = ap.parse_args()
    if a.cond != "A" and a.s_star is None:
        raise SystemExit("--cond C 계열은 --s-star 가 필요합니다 (C 0.7985, C′ 0.5982)")
    from ultralytics import YOLO
    model = YOLO(a.model)
    ref = json.loads(REF.read_text(encoding="utf-8")) if REF.exists() else {"A": {"n_pred": [0.0] * 101, "sub": [0.0] * 101}}
    p = Path(a.image)
    images = sorted([q for q in p.iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg")]) if p.is_dir() else [p]
    out = Path(a.out)
    for q in images:
        info = run_one(model, q, a.diameter, a.cond, a.s_star, a.T, a.overlap, a.wrap, a.conf, a.lam, a.s_range, a.review_top, ref, out)
        print(json.dumps({k: v for k, v in info.items() if k != "detections"}, ensure_ascii=False))
    print("저장:", out.resolve())


if __name__ == "__main__":
    main()
