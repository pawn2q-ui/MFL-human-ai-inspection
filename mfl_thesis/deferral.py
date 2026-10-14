"""
AI 판단보류 (4.3–4.4절): 영상별 검토 우선순위 점수 -> 상위 k% 보류 -> 잔여 미탐 평가.

점수 후보 (--score)
  sub_thr_max   : 운영 임계값(0.25) 아래 원시 검출의 최대 신뢰도 ("거의 잡힐 뻔한 신호"). 없으면 0  [기본]
  n_pred        : 운영 임계값 이상 검출 수 (결함 밀집 영상 우선; v0.2 추가)
  combo         : 배관 안에서 n_pred 순위 백분율 + sub_thr_max 순위 백분율 (v0.2 추가, 영상 기준 최고)
  one_minus_max : 1 - (영상의 최대 신뢰도). 검출이 전혀 없으면 1 (보수적)
  ensemble      : 시드가 다른 두 모델의 예측 불일치 (--preds2 필요)
  knn           : 특징 거리 (--features npz: stems, X; --train-features npz) 선택
  random        : 무작위 (비교 기준, 내부적으로 반복)
관경 외삽 항: e_p = max(0, s_y-s_max, s_min-s_y)/(s_max-s_min), 최종 점수 = u + lambda*e_p

사용: python -m mfl_thesis.deferral --manifest manifest.csv --preds preds/C_s0/preds --split test --out defer/E_s0 \
          --score sub_thr_max --lambda 1.0 --ks 5,10,20
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import IMG_W, IMG_H, DAMAGE_IDS, A_CONF, s_y
from .labels import read_yolo, read_pred, iou_matrix
from .manifest import read_manifest
from .evaluate import match_image


def train_s_range(rows):
    s = [s_y(r["diameter_mm"]) for r in rows if r["split"] == "train"]
    return min(s), max(s)


def extrapolation_term(D, s_min, s_max):
    s = s_y(D)
    return max(0.0, s - s_max, s_min - s) / max(1e-9, s_max - s_min)


def ensemble_disagreement(p1, p2, thr=A_CONF, iou_thr=0.5):
    """두 모델의 운영 임계값 이상 예측 간 불일치: 짝 없는 박스 비율 + 짝지은 박스의 신뢰도 차 평균."""
    a = p1[p1[:, 1] >= thr] if len(p1) else p1; b = p2[p2[:, 1] >= thr] if len(p2) else p2
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 1.0
    ious = iou_matrix(a[:, 2:6], b[:, 2:6])
    same = a[:, 0][:, None] == b[:, 0][None, :]
    ious = ious * same
    matched_a = ious.max(1) >= iou_thr; matched_b = ious.max(0) >= iou_thr
    unmatched = ((~matched_a).sum() + (~matched_b).sum()) / (len(a) + len(b))
    conf_diff = np.mean([abs(a[i, 1] - b[ious[i].argmax(), 1]) for i in np.where(matched_a)[0]]) if matched_a.any() else 0.0
    return float(min(1.0, unmatched + conf_diff))


def image_table(rows, preds_dir, preds2_dir=None, thr=A_CONF, features=None, train_features=None, knn_k=5):
    preds_dir = Path(preds_dir)
    recs = []
    for r in rows:
        gt = read_yolo(r["label"], IMG_W, IMG_H) if r["label"] else np.zeros((0, 5))
        gt_d = gt[np.isin(gt[:, 0], DAMAGE_IDS)] if len(gt) else gt
        pr = read_pred(preds_dir / f"{r['stem']}.txt")
        pr_op = pr[pr[:, 1] >= thr] if len(pr) else pr
        hit, _, _ = match_image(gt_d, pr_op)
        sub = pr[pr[:, 1] < thr][:, 1] if len(pr) else np.zeros(0)
        rec = {"stem": r["stem"], "pipe": r["pipe_id"], "D": r["diameter_mm"], "n_gt": int(len(gt_d)),
               "n_miss": int((~hit).sum()), "n_pred_thr": int(len(pr_op)),
               "sub_thr_max": float(sub.max()) if len(sub) else 0.0,
               "one_minus_max": float(1 - pr[:, 1].max()) if len(pr) else 1.0}
        if preds2_dir is not None:
            pr2 = read_pred(Path(preds2_dir) / f"{r['stem']}.txt")
            rec["ensemble"] = ensemble_disagreement(pr, pr2, thr)
        recs.append(rec)
    df = pd.DataFrame(recs)
    df["n_pred"] = df["n_pred_thr"].astype(float)
    # combo: 배관별 순위 백분율 합 (라벨 불필요)
    def _rank_pct(x):
        order = np.argsort(np.argsort(x.values, kind="stable"), kind="stable")
        return pd.Series((order + 1) / max(1, len(x)), index=x.index)
    df["combo"] = df.groupby("pipe")["n_pred_thr"].transform(_rank_pct) + df.groupby("pipe")["sub_thr_max"].transform(_rank_pct)
    if features is not None and train_features is not None:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=knn_k).fit(train_features["X"])
        fmap = {s: x for s, x in zip(features["stems"], features["X"])}
        X = np.stack([fmap[s] for s in df.stem])
        d, _ = nn.kneighbors(X)
        df["knn"] = d.mean(1)
    return df


def evaluate_deferral(df, score_col, lam, s_min, s_max, ks=(5, 10, 20), n_random=100, r_list=(0.5, 0.75, 1.0), seed=0):
    df = df.copy()
    df["e_p"] = df.D.map(lambda D: extrapolation_term(D, s_min, s_max))
    df["u"] = df[score_col] if score_col != "random" else 0.0
    df["score"] = df["u"] + lam * df["e_p"]
    total_gt = df.n_gt.sum(); total_miss = df.n_miss.sum()
    rng = np.random.default_rng(seed)
    order = np.argsort(-df.score.values, kind="stable")
    res = {"score": score_col, "lambda": lam, "n_images": int(len(df)), "total_gt": int(total_gt), "total_miss": int(total_miss),
           "baseline_miss_rate": float(total_miss / max(total_gt, 1)), "k": {}}
    # AUROC: 점수가 '미탐이 있는 영상'을 얼마나 잘 분리하는가
    y = (df.n_miss > 0).values
    if y.any() and (~y).any():
        from sklearn.metrics import roc_auc_score
        res["auroc_has_miss"] = float(roc_auc_score(y, df.score.values))
    for k in ks:
        n_def = int(round(len(df) * k / 100))
        def summarize(idx_def):
            mask = np.zeros(len(df), dtype=bool); mask[idx_def] = True
            miss_def = df.n_miss.values[mask].sum(); miss_auto = df.n_miss.values[~mask].sum()
            gt_auto = df.n_gt.values[~mask].sum()
            out = {"residual_miss_rate": miss_auto / max(total_gt, 1),            # 전체 손상 대비 자동 영역 잔여 미탐
                   "auto_miss_rate": miss_auto / max(gt_auto, 1),                 # 자동 영역 내부 미탐률
                   "capture_rate": miss_def / max(total_miss, 1),                 # 보류로 포착한 미탐 비율
                   "boxes_to_review": int(df.n_pred_thr.values[mask].sum()),
                   "per_pipe_deferred_frac": df[mask].groupby("pipe").size().div(df.groupby("pipe").size()).fillna(0).round(3).to_dict()}
            for r in r_list:
                out[f"residual_r{int(r*100)}"] = (miss_auto + (1 - r) * miss_def) / max(total_gt, 1)
            return out
        chosen = summarize(order[:n_def])
        rand = [summarize(rng.choice(len(df), n_def, replace=False)) for _ in range(n_random)]
        rand_stats = {key: (float(np.mean([x[key] for x in rand])), float(np.std([x[key] for x in rand])))
                      for key in ["residual_miss_rate", "capture_rate", "auto_miss_rate"]}
        res["k"][str(k)] = {"n_deferred": n_def, "score_based": chosen, "random_mean_std": rand_stats}
    # Risk–Coverage 곡선 (k 0..50%)
    curve = []
    for k in range(0, 51):
        n_def = int(round(len(df) * k / 100)); mask = np.zeros(len(df), dtype=bool); mask[order[:n_def]] = True
        curve.append({"review_pct": k, "coverage": 1 - n_def / len(df), "residual_miss_rate": df.n_miss.values[~mask].sum() / max(total_gt, 1)})
    res["risk_coverage"] = curve
    return res, df


def plot_risk_coverage(results, out_png):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, res in results.items():
        c = res["risk_coverage"]
        ax.plot([x["review_pct"] for x in c], [x["residual_miss_rate"] for x in c], label=name)
    ax.set_xlabel("images referred for human review (%)"); ax.set_ylabel("residual miss rate (of all damage GT)")
    ax.legend(); fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--preds2", default=None, help="앙상블 점수용 두 번째 모델 예측 폴더")
    ap.add_argument("--split", default="test")
    ap.add_argument("--score", default="sub_thr_max", choices=["sub_thr_max", "n_pred", "combo", "one_minus_max", "ensemble", "knn", "random"])
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--thr", type=float, default=A_CONF)
    ap.add_argument("--features", default=None); ap.add_argument("--train-features", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows_all = read_manifest(a.manifest)
    rows = [r for r in rows_all if r["split"] in set(a.split.split(","))]
    s_min, s_max = train_s_range(rows_all)
    feats = np.load(a.features) if a.features else None
    tfeats = np.load(a.train_features) if a.train_features else None
    df = image_table(rows, a.preds, a.preds2, a.thr, feats, tfeats)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for lam in sorted({0.0, a.lam}):
        res, dfk = evaluate_deferral(df, a.score, lam, s_min, s_max, tuple(int(k) for k in a.ks.split(",")))
        results[f"{a.score}_lambda{lam:g}"] = res
        dfk.to_csv(out / f"image_scores_lambda{lam:g}.csv", index=False)
    (out / "deferral.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=float))
    plot_risk_coverage(results, out / "risk_coverage.png")
    for name, res in results.items():
        print(name, {k: (round(float(v["score_based"]["residual_miss_rate"]), 4), round(float(v["random_mean_std"]["residual_miss_rate"][0]), 4)) for k, v in res["k"].items()})


if __name__ == "__main__":
    main()
