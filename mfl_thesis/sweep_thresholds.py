"""
신뢰도 임계값 스윕: 조건별로 임계값을 바꿔 가며 MTL 재현율·정밀도·오탐을 전체/배관/457 그룹 단위로 집계한다.
'같은 정밀도에서의 재현율 비교'(검증 배관 F 에서 정밀도가 목표값이 되는 임계값을 골라 시험 배관에 적용)에 쓴다.

  python -m mfl_thesis.sweep_thresholds --manifest manifest.csv --split val ^
      --preds preds/A_s0/preds,preds/B_s0/preds --names A_s0,B_s0 --thr 0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6 --out analysis/sweep_val.csv
출력 CSV 열: name, split, thr, scope(overall / 배관 ID / G457), cls, n_gt, recall, precision, fp
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .config import A_CONF
from .evaluate import build_records, class_metrics
from .manifest import read_manifest

GROUPS = {"G457": ["A", "B", "C"]}


def sweep(manifest, preds_dirs, names, splits, thrs, out, classes=("MTL", "SWA"), verbose=True):
    rows_all = read_manifest(manifest)
    rows = [r for r in rows_all if r["split"] in splits]
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    for name, pdir in zip(names, preds_dirs):
        for thr in thrs:
            gt, pr = build_records(rows, pdir, conf_thr=thr, ap_conf=thr)
            if pr.empty or "pipe" not in pr.columns:     # 높은 임계값에서 예측이 하나도 없을 때
                pr = pd.DataFrame(columns=["stem", "pipe", "cls_id", "cls", "conf", "tp"])
            scopes = {"overall": (gt, pr)}
            for pipe, g in gt.groupby("pipe"):
                scopes[pipe] = (g, pr[pr["pipe"] == pipe])
            for gname, pipes in GROUPS.items():
                g = gt[gt["pipe"].isin(pipes)]
                if len(g):
                    scopes[gname] = (g, pr[pr["pipe"].isin(pipes)])
            for scope, (g, p) in scopes.items():
                m = class_metrics(g, p, thr)
                for cls in classes:
                    if cls not in m:
                        continue
                    recs.append({"name": name, "split": "+".join(sorted(splits)), "thr": thr, "scope": scope, "cls": cls,
                                 "n_gt": m[cls]["n_gt"], "recall": m[cls]["recall"], "precision": m[cls]["precision"], "fp": m[cls]["fp"]})
            if verbose:
                o = [x for x in recs if x["name"] == name and x["thr"] == thr and x["scope"] == "overall" and x["cls"] == "MTL"][0]
                print(f"{name} thr={thr:.2f}: MTL recall={o['recall']:.3f} precision={o['precision']:.3f} fp={o['fp']}")
        pd.DataFrame(recs).to_csv(out.with_suffix('.partial.csv'), index=False)   # 조건마다 중간 저장(완료 표시는 최종 파일)
    pd.DataFrame(recs).to_csv(out, index=False)
    if verbose:
        print("저장:", out)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--preds", required=True, help="쉼표 구분 preds/ 폴더들")
    ap.add_argument("--names", required=True, help="쉼표 구분 이름들 (preds 와 같은 수)")
    ap.add_argument("--thr", default="0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    preds = [p.strip() for p in a.preds.split(",") if p.strip()]
    names = [n.strip() for n in a.names.split(",") if n.strip()]
    assert len(preds) == len(names), "--preds 와 --names 의 개수가 다릅니다"
    thrs = [float(t) for t in a.thr.split(",")]
    sweep(a.manifest, preds, names, set(a.split.split(",")), thrs, a.out)


if __name__ == "__main__":
    main()
