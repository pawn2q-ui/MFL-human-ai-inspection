"""시연 프로그램 점검: 매니페스트에서 배관별 양성 시험 영상 n장을 골라 demo.run_one 을 돌리고 결과 그림·JSON을 남긴다.
  python -m mfl_thesis.demo_check --manifest manifest.csv --model runs/A_s1/weights/best.pt --pipes A,B,J --n 2 --out demo_out
발표 때는 demo_out/<stem>_pred.jpg 를 그대로 보여 주거나, 같은 영상으로 `python -m mfl_thesis.demo ...` 를 현장에서 실행한다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .config import IMG_W, IMG_H, DAMAGE_IDS
from .labels import read_yolo
from .manifest import read_manifest
from .demo import run_one, REF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--pipes", default="A,B,J"); ap.add_argument("--n", type=int, default=2); ap.add_argument("--split", default="test")
    ap.add_argument("--cond", default="A"); ap.add_argument("--s-star", type=float, default=None)
    ap.add_argument("--out", default="demo_out")
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.model)
    ref = json.loads(REF.read_text(encoding="utf-8"))
    rows = [r for r in read_manifest(a.manifest) if r["split"] == a.split]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = []
    for pid in a.pipes.split(","):
        picked = 0
        for r in rows:
            if r["pipe_id"] != pid:
                continue
            gt = read_yolo(r["label"], IMG_W, IMG_H) if r["label"] else np.zeros((0, 5))
            if len(gt) == 0 or not np.isin(gt[:, 0], DAMAGE_IDS).any():
                continue
            info = run_one(model, Path(r["image"]), float(r["diameter_mm"]), a.cond, a.s_star, 1280, 0.458, 1, 0.25, 0.0, "0.7985,1.0640", 20.0, ref, out)
            info["n_gt_damage"] = int(np.isin(gt[:, 0], DAMAGE_IDS).sum())
            summary.append({k: v for k, v in info.items() if k != "detections"})
            print(pid, r["stem"], "정답 손상", info["n_gt_damage"], "검출", info["n_pred"], "검토 권고", info["review_recommended"], f"{info['elapsed_s']} s")
            picked += 1
            if picked >= a.n:
                break
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "DONE").write_text("ok")
    print("저장:", out.resolve())


if __name__ == "__main__":
    main()
