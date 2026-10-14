"""
오탐(FP) 근접 분석: 운영 임계값 이상 예측 가운데 정답과 대응되지 않은 박스(FP)가 정답과 얼마나 가까운지 본다.
타일링 조건에서 정밀도가 낮은 이유가 "정답 근처의 중복·국소화 불량 검출"인지, "정답과 무관한 검출"인지 가른다.

  python -m mfl_thesis.fp_analysis --manifest manifest.csv --split test ^
      --preds preds/A_s1/preds,preds/B_s1/preds --names A_s1,B_s1 --out analysis/fp_analysis.csv
출력 열: name, scope, n_fp, frac_iou0 (정답과 전혀 안 겹침), frac_iou_0_01, frac_iou_01_03, frac_iou_03_05 (근접 오탐),
        frac_dup (다른 예측과 IoU>=0.3 — NMS 뒤에도 남은 중복), frac_other_cls (다른 클래스 정답과 IoU>=0.5), median_conf
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import IMG_W, IMG_H, A_CONF, DAMAGE_IDS
from .labels import read_yolo, read_pred, iou_matrix
from .manifest import read_manifest
from .evaluate import match_image

GROUPS = {"G457": ["A", "B", "C"]}


def analyze(manifest, preds_dirs, names, splits, out, thr=A_CONF, verbose=True):
    rows = [r for r in read_manifest(manifest) if r["split"] in splits]
    recs = []
    for name, pdir in zip(names, preds_dirs):
        pdir = Path(pdir)
        fps = []   # (pipe, conf, max_iou_same_cls, max_iou_any, dup)
        for r in rows:
            gt = read_yolo(r["label"], IMG_W, IMG_H) if r["label"] else np.zeros((0, 5))
            pr = read_pred(pdir / f"{r['stem']}.txt")
            pr = pr[pr[:, 1] >= thr] if len(pr) else pr
            if len(pr) == 0:
                continue
            _, _, tp = match_image(gt, pr)          # evaluate 와 같은 클래스별 일대일 대응(12종 전체)
            fp_idx = np.where(~tp)[0]
            if len(fp_idx) == 0:
                continue
            iou_gt = iou_matrix(pr[:, 2:6], gt[:, 1:5]) if len(gt) else np.zeros((len(pr), 0))
            iou_pp = iou_matrix(pr[:, 2:6], pr[:, 2:6]) if len(pr) > 1 else np.zeros((len(pr), len(pr)))
            np.fill_diagonal(iou_pp, 0)
            for i in fp_idx:
                same = (gt[:, 0] == pr[i, 0]) if len(gt) else np.zeros(0, bool)
                mi_same = float(iou_gt[i][same].max()) if same.any() else 0.0
                mi_any = float(iou_gt[i].max()) if iou_gt.shape[1] else 0.0
                other = float(iou_gt[i][~same].max()) if (~same).any() else 0.0
                fps.append((r["pipe_id"], float(pr[i, 1]), mi_same, mi_any, float(iou_pp[i].max()), other))
        df = pd.DataFrame(fps, columns=["pipe", "conf", "iou_same", "iou_any", "iou_pp", "iou_other"])
        scopes = {"overall": df}
        for p, g in df.groupby("pipe"):
            scopes[p] = g
        for gname, pipes in GROUPS.items():
            scopes[gname] = df[df["pipe"].isin(pipes)]
        for scope, g in scopes.items():
            if len(g) == 0:
                continue
            recs.append({"name": name, "scope": scope, "n_fp": int(len(g)),
                         "frac_iou0": float((g.iou_same == 0).mean()),
                         "frac_iou_0_01": float(((g.iou_same > 0) & (g.iou_same < 0.1)).mean()),
                         "frac_iou_01_03": float(((g.iou_same >= 0.1) & (g.iou_same < 0.3)).mean()),
                         "frac_iou_03_05": float(((g.iou_same >= 0.3) & (g.iou_same < 0.5)).mean()),
                         "frac_dup": float((g.iou_pp >= 0.3).mean()),
                         "frac_other_cls": float((g.iou_other >= 0.5).mean()),
                         "median_conf": float(g.conf.median())})
        if verbose:
            o = [x for x in recs if x["name"] == name and x["scope"] == "overall"][0]
            print(f"{name}: FP {o['n_fp']} | 정답과 무관 {o['frac_iou0']:.2f} | 근접(0.1–0.5) {o['frac_iou_01_03'] + o['frac_iou_03_05']:.2f} | 중복(IoU_pp>=0.3) {o['frac_dup']:.2f} | 다른 클래스 정답 {o['frac_other_cls']:.2f}")
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_csv(out, index=False)
    if verbose:
        print("저장:", out)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--thr", type=float, default=A_CONF)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    preds = [p.strip() for p in a.preds.split(",") if p.strip()]
    names = [n.strip() for n in a.names.split(",") if n.strip()]
    assert len(preds) == len(names)
    analyze(a.manifest, preds, names, set(a.split.split(",")), a.out, a.thr)


if __name__ == "__main__":
    main()
