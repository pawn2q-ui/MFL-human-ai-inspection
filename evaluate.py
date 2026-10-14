"""
탐지 성능 평가 (3.4절)
  - 정답-예측 일대일 대응(클래스 동일, IoU>=0.5, 신뢰도 내림차순 탐욕) -> 박스별 hit/miss
  - 손상 클래스별 Recall·미탐률·Precision·FP·AP50 (운영 임계값 0.25)
  - 결함 실제 길이(mm)에 대한 POD 곡선(로지스틱, log 길이) + a90 (+ 부트스트랩 CI), 구간별 POD(Wilson 신뢰대)
  - mm 구간별 미탐률, 원주 이음부 결함 미탐률
  - 두 조건 간 쌍체 부트스트랩 신뢰구간

사용: python -m mfl_thesis.evaluate --manifest manifest.csv --preds preds/C_s0/preds --split test --out eval/C_s0
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import IMG_W, IMG_H, CLASSES, DAMAGE, DAMAGE_IDS, MATCH_IOU, A_CONF, s_y
from .labels import read_yolo, read_pred, iou_matrix
from .manifest import read_manifest

SEAM_PX = 2
MM_BINS = [0, 10, 20, 40, 80, np.inf]
MM_LABELS = ["<10", "10-20", "20-40", "40-80", ">=80"]


# ----------------------------------------------------------------------------
# 대응
# ----------------------------------------------------------------------------
def match_image(gt, pr, iou_thr=MATCH_IOU):
    """gt (N,5), pr (M,6, conf 내림차순 정렬 안 되어 있어도 됨) -> gt_hit (N,), gt_conf (N,), pr_tp (M,)"""
    gt_hit = np.zeros(len(gt), dtype=bool); gt_conf = np.full(len(gt), np.nan)
    pr_tp = np.zeros(len(pr), dtype=bool)
    if len(pr) == 0 or len(gt) == 0:
        return gt_hit, gt_conf, pr_tp
    order = np.argsort(-pr[:, 1])
    for c in np.unique(pr[:, 0]):
        g_idx = np.where(gt[:, 0] == c)[0]
        if len(g_idx) == 0:
            continue
        p_idx = [i for i in order if pr[i, 0] == c]
        ious = iou_matrix(pr[p_idx, 2:6], gt[g_idx, 1:5])
        used = np.zeros(len(g_idx), dtype=bool)
        for k, i in enumerate(p_idx):
            cand = np.where((ious[k] >= iou_thr) & ~used)[0]
            if len(cand):
                j = cand[np.argmax(ious[k, cand])]
                used[j] = True; pr_tp[i] = True
                gt_hit[g_idx[j]] = True; gt_conf[g_idx[j]] = pr[i, 1]
    return gt_hit, gt_conf, pr_tp


def voc_ap(tp_sorted, n_gt):
    """tp_sorted: 신뢰도 내림차순 TP 플래그 -> AP(연속 보간, VOC2010+)"""
    if n_gt == 0 or len(tp_sorted) == 0:
        return float("nan")
    tp = np.cumsum(tp_sorted); fp = np.cumsum(~tp_sorted)
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# ----------------------------------------------------------------------------
# 레코드 생성
# ----------------------------------------------------------------------------
def build_records(manifest_rows, preds_dir, conf_thr=A_CONF, ap_conf=A_CONF):
    """-> gt_df (박스별), pred_df (예측별, conf>=ap_conf)"""
    preds_dir = Path(preds_dir)
    gt_rows, pr_rows = [], []
    for r in manifest_rows:
        gt = read_yolo(r["label"], IMG_W, IMG_H) if r["label"] else np.zeros((0, 5))
        pr_all = read_pred(preds_dir / f"{r['stem']}.txt")
        pr_ap = pr_all[pr_all[:, 1] >= ap_conf] if len(pr_all) else pr_all
        pr_op = pr_all[pr_all[:, 1] >= conf_thr] if len(pr_all) else pr_all
        hit_op, conf_op, _ = match_image(gt, pr_op)
        _, _, tp_ap = match_image(gt, pr_ap)
        sy = s_y(r["diameter_mm"])
        for i, (c, x1, y1, x2, y2) in enumerate(gt):
            w, h = x2 - x1, y2 - y1
            gt_rows.append({"stem": r["stem"], "pipe": r["pipe_id"], "D": r["diameter_mm"], "split": r["split"], "gi": i,
                            "cls": CLASSES[int(c)], "cls_id": int(c), "w_px": w, "h_px": h,
                            "L_circ_mm": h * sy, "L_ax_mm": w * 1.0,
                            "seam": bool(y1 <= SEAM_PX or y2 >= IMG_H - SEAM_PX),
                            "hit": bool(hit_op[i]), "conf": float(conf_op[i]) if not np.isnan(conf_op[i]) else np.nan})
        for i, p in enumerate(pr_ap):
            pr_rows.append({"stem": r["stem"], "pipe": r["pipe_id"], "cls_id": int(p[0]), "cls": CLASSES[int(p[0])],
                            "conf": float(p[1]), "tp": bool(tp_ap[i])})
    return pd.DataFrame(gt_rows), pd.DataFrame(pr_rows)


# ----------------------------------------------------------------------------
# 지표
# ----------------------------------------------------------------------------
def class_metrics(gt: pd.DataFrame, pr: pd.DataFrame, conf_thr=A_CONF, classes=DAMAGE):
    out = {}
    for c in classes:
        g = gt[gt.cls == c]; p = pr[(pr.cls == c)]
        p_op = p[p.conf >= conf_thr]
        n_gt, n_hit = len(g), int(g.hit.sum())
        tp = int(p_op.tp.sum()); fp = int((~p_op.tp).sum())
        p_sorted = p.sort_values("conf", ascending=False)
        out[c] = {"n_gt": n_gt, "recall": n_hit / n_gt if n_gt else np.nan, "miss_rate": 1 - n_hit / n_gt if n_gt else np.nan,
                  "precision": tp / (tp + fp) if (tp + fp) else np.nan, "fp": fp, "ap50": voc_ap(p_sorted.tp.values, n_gt)}
    d = gt[gt.cls.isin(classes)]
    out["DMG_pooled"] = {"n_gt": len(d), "recall": d.hit.mean() if len(d) else np.nan,
                         "miss_rate": 1 - d.hit.mean() if len(d) else np.nan,
                         "mAP50": float(np.nanmean([out[c]["ap50"] for c in classes]))}
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den; half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def fit_pod(lengths_mm, hits, n_boot=200, seed=0):
    """로지스틱 POD(a): logit P = b0 + b1*log a. 반환 dict (b0,b1,a90, a90 CI)."""
    from sklearn.linear_model import LogisticRegression
    L = np.asarray(lengths_mm, float); y = np.asarray(hits, bool)
    ok = L > 0
    L, y = L[ok], y[ok]
    res = {"n": int(len(L)), "b0": np.nan, "b1": np.nan, "a90": np.nan, "a90_lo": np.nan, "a90_hi": np.nan}
    if len(L) < 10 or y.all() or (~y).all():
        return res
    x = np.log(L)[:, None]

    def _fit(xx, yy):
        m = LogisticRegression(C=1e9, max_iter=1000).fit(xx, yy)
        b1, b0 = float(m.coef_[0, 0]), float(m.intercept_[0])
        a90 = np.exp((np.log(0.9 / 0.1) - b0) / b1) if b1 > 0 else np.nan
        return b0, b1, a90
    res["b0"], res["b1"], res["a90"] = _fit(x, y)
    if n_boot:
        rng = np.random.default_rng(seed); a90s = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(L), len(L))
            if y[idx].all() or (~y[idx]).all():
                continue
            a90s.append(_fit(x[idx], y[idx])[2])
        a90s = np.array([v for v in a90s if np.isfinite(v)])
        if len(a90s) > 10:
            res["a90_lo"], res["a90_hi"] = np.nanpercentile(a90s, [2.5, 97.5])
    return res


def pod_table(gt: pd.DataFrame, length_col="L_circ_mm", classes=DAMAGE, n_boot=200):
    rows = []
    groups = [("ALL", gt)] + [(p, g) for p, g in gt.groupby("pipe")]
    for pipe, g in groups:
        for c in list(classes) + ["DMG_pooled"]:
            gg = g[g.cls.isin(classes)] if c == "DMG_pooled" else g[g.cls == c]
            r = fit_pod(gg[length_col], gg.hit, n_boot)
            rows.append({"pipe": pipe, "D": g.D.iloc[0] if pipe != "ALL" else np.nan, "cls": c, "length": length_col, **r})
    return pd.DataFrame(rows)


def binned_pod(gt: pd.DataFrame, length_col="L_circ_mm", classes=DAMAGE, bins=MM_BINS, labels=MM_LABELS):
    d = gt[gt.cls.isin(classes)].copy()
    d["bin"] = pd.cut(d[length_col], bins=bins, labels=labels)
    rows = []
    for (pipe, b), g in d.groupby(["pipe", "bin"], observed=True):
        k, n = int(g.hit.sum()), len(g); lo, hi = wilson(k, n)
        rows.append({"pipe": pipe, "bin": b, "n": n, "pod": k / n, "miss_rate": 1 - k / n, "ci_lo": lo, "ci_hi": hi})
    for b, g in d.groupby("bin", observed=True):
        k, n = int(g.hit.sum()), len(g); lo, hi = wilson(k, n)
        rows.append({"pipe": "ALL", "bin": b, "n": n, "pod": k / n, "miss_rate": 1 - k / n, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def seam_metrics(gt: pd.DataFrame, classes=DAMAGE):
    d = gt[gt.cls.isin(classes)]
    s, ns = d[d.seam], d[~d.seam]
    return {"seam_n": len(s), "seam_miss_rate": 1 - s.hit.mean() if len(s) else np.nan,
            "nonseam_n": len(ns), "nonseam_miss_rate": 1 - ns.hit.mean() if len(ns) else np.nan}


def paired_bootstrap(gt_a: pd.DataFrame, gt_b: pd.DataFrame, classes=DAMAGE, n=1000, seed=0):
    """같은 정답 박스 집합에 대한 두 조건(a,b)의 미탐률 차이(a-b)의 95% CI (영상 단위 재표집)."""
    a = gt_a[gt_a.cls.isin(classes)].set_index(["stem", "gi"]).sort_index()
    b = gt_b[gt_b.cls.isin(classes)].set_index(["stem", "gi"]).sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    stems = np.array(sorted({s for s, _ in common}))
    a_by = a.groupby(level=0).hit.agg(["sum", "size"]); b_by = b.groupby(level=0).hit.agg(["sum", "size"])
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(n):
        pick = rng.choice(stems, len(stems), replace=True)
        na, nb = a_by.loc[pick], b_by.loc[pick]
        tot = na["size"].sum()
        if tot == 0:
            continue
        diffs.append((1 - na["sum"].sum() / tot) - (1 - nb["sum"].sum() / tot))
    diffs = np.array(diffs)
    point = (1 - a.hit.mean()) - (1 - b.hit.mean())
    return {"miss_diff_a_minus_b": float(point), "ci_lo": float(np.percentile(diffs, 2.5)), "ci_hi": float(np.percentile(diffs, 97.5)), "n_boxes": int(len(common))}


def plot_pod(gt: pd.DataFrame, pod: pd.DataFrame, out_png, length_col="L_circ_mm", classes=DAMAGE):
    d = gt[gt.cls.isin(classes)]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = np.linspace(max(1.0, d[length_col].quantile(0.01)), d[length_col].quantile(0.99), 200)
    for pipe, g in d.groupby("pipe"):
        r = pod[(pod["pipe"] == pipe) & (pod.cls == "DMG_pooled")].iloc[0]
        if np.isfinite(r.b1):
            p = 1 / (1 + np.exp(-(r.b0 + r.b1 * np.log(xs))))
            lab = f"{pipe} ({g.D.iloc[0]:.0f} mm), a90={r.a90:.0f} mm" if np.isfinite(r.a90) else f"{pipe} ({g.D.iloc[0]:.0f} mm)"
            ax.plot(xs, p, label=lab)
    ax.axhline(0.9, ls=":", c="gray"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("defect circumferential length (mm)"); ax.set_ylabel("POD")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def evaluate(manifest, preds_dir, splits, out, conf_thr=A_CONF, ap_conf=A_CONF, n_boot=200, verbose=True, pipes=None):
    rows = [r for r in read_manifest(manifest) if r["split"] in splits and (pipes is None or r["pipe_id"] in pipes)]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    gt, pr = build_records(rows, preds_dir, conf_thr, ap_conf)
    metrics = {"overall": class_metrics(gt, pr, conf_thr)}
    for pipe, g in gt.groupby("pipe"):
        metrics[f"pipe_{pipe}"] = class_metrics(g, pr[pr["pipe"] == pipe], conf_thr)
    metrics["seam"] = seam_metrics(gt)
    pod = pod_table(gt, "L_circ_mm", n_boot=n_boot)
    pod_ax = pod_table(gt, "L_ax_mm", n_boot=0)
    bins = binned_pod(gt)
    gt.to_csv(out / "gt_records.csv", index=False); pr.to_csv(out / "pred_records.csv", index=False)
    pod.to_csv(out / "pod_circ.csv", index=False); pod_ax.to_csv(out / "pod_axial.csv", index=False)
    bins.to_csv(out / "pod_bins_circ.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=float))
    plot_pod(gt, pod, out / "pod_circ.png")
    if verbose:
        o = metrics["overall"]
        print(f"[{out.name}] DMG pooled: recall={o['DMG_pooled']['recall']:.3f} miss={o['DMG_pooled']['miss_rate']:.3f} mAP50={o['DMG_pooled']['mAP50']:.3f}")
        print(pod[pod.cls == "DMG_pooled"][["pipe", "D", "n", "a90", "a90_lo", "a90_hi"]].round(1).to_string(index=False))
    return gt, pr, metrics, pod, bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--preds", required=True, help="preds/ 폴더 (영상별 txt)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf-thr", type=float, default=A_CONF)
    ap.add_argument("--ap-conf", type=float, default=A_CONF, help="AP 계산에 쓸 최소 신뢰도 (공식=0.25)")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--pipes", default=None, help="쉼표 구분 배관 ID 로 제한 (예: A,B,C)")
    a = ap.parse_args()
    evaluate(a.manifest, a.preds, set(a.split.split(",")), a.out, a.conf_thr, a.ap_conf, a.n_boot,
             pipes=set(a.pipes.split(",")) if a.pipes else None)


if __name__ == "__main__":
    main()
