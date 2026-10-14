"""
5장 그림 생성 (eval/, defer/ 산출물로부터). 2026-10-10
  python -m mfl_thesis.ch5_figs --eval eval --defer defer --out analysis/ch5
그림 5-1: 관경 그룹별 MTL 원주 길이 구간 POD, 조건별 (두 시드 평균)
그림 5-2: 배관별 MTL 재현율, 조건별 (막대 = 두 시드 평균, 점 = 각 시드)
그림 5-3: 조건 C 모델의 추론 배율 실험 (457 mm 배관) — 재현율·정밀도
그림 5-4: 판단보류 Risk–Coverage 곡선 (조건 D·E·E⁻·E′, 점선 = 무작위 보류 기대값)
그림 5-5: 소수 표본 적응 전후 (재현율·AP50·SWA 재현율)
표 CSV 도 함께 저장한다 (tab5_*.csv).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for name in ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK SC", "AppleGothic"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.linewidth": 0.8, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "grid.color": "#e6e5e1", "grid.linewidth": 0.6})

# 조건별 색(고정 순서) · 마커(색 외 2차 부호화)
COND = {"A": ("A 공식 프로토콜", "#2a78d6", "o"), "B": ("B 동일 배율", "#eb6834", "s"), "C": ("C 관경 보정", "#1baf7a", "^"),
        "C⁻": ("C− 축소 금지 추론", "#eda100", "v"), "C′": ("C′ 축소 없는 보정", "#e87ba4", "D")}   # 그림 안에서는 ⁻ 대신 − (글꼴 호환)
RUNS = {"A": ("A_s0", "A_s1"), "B": ("B_s0", "B_s1"), "C": ("C_s0", "C_s1"), "C⁻": ("C_s0_clamp", "C_s1_clamp"), "C′": ("Cp_s0", "Cp_s1")}
GROUPS = {"457 mm (A·B·C)": ["A", "B", "C"], "508 mm (D)": ["D"], "813 mm (J)": ["J"]}
PIPE_LABEL = {"A": "A\n457", "B": "B\n457", "C": "C\n457", "D": "D\n508", "J": "J\n813"}
BINS = [0, 20, 40, 80, 160, np.inf]
BIN_LABELS = ["≤20", "20–40", "40–80", "80–160", ">160"]
GRAY, INK, INK2 = "#8a8984", "#0b0b0b", "#52514e"


def gt_mtl(eval_dir: Path, run: str) -> pd.DataFrame:
    d = pd.read_csv(eval_dir / run / "gt_records.csv")
    d = d[d["cls"] == "MTL"].copy()
    d["bin"] = pd.cut(d["L_circ_mm"], BINS, labels=BIN_LABELS, right=True)
    return d


def metrics(eval_dir: Path, run: str) -> dict:
    return json.loads((eval_dir / run / "metrics.json").read_text(encoding="utf-8"))


def fig5_1(eval_dir: Path, out: Path, min_n=30):
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0), dpi=200, sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (gname, pipes) in zip(axes, GROUPS.items()):
        for cond, (label, color, marker) in COND.items():
            ys = []
            for lab in BIN_LABELS:
                vals, n = [], 0
                for run in RUNS[cond]:
                    d = gt_mtl(eval_dir, run)
                    d = d[d["pipe"].isin(pipes) & (d["bin"] == lab)]
                    n = len(d); vals.append(d["hit"].mean() if n else np.nan)
                y = np.nanmean(vals) if n >= min_n else np.nan
                ys.append(y); rows.append({"group": gname, "cond": cond, "bin": lab, "n": n, "pod": y})
            ax.plot(range(len(BIN_LABELS)), ys, color=color, marker=marker, ms=4, lw=1.6, label=label, mec="white", mew=0.6)
        ax.set_xticks(range(len(BIN_LABELS))); ax.set_xticklabels(BIN_LABELS, fontsize=7.5)
        ax.set_title(gname); ax.set_ylim(0, 0.8); ax.set_xlim(-0.4, len(BIN_LABELS) - 0.6); ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.set_xlabel("MTL 원주 방향 길이 (mm)")
    axes[0].set_ylabel("탐지확률 POD (신뢰도 0.25)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=5, frameon=False, handlelength=1.6, columnspacing=1.0)
    fig.savefig(out / "fig5-1_pod_bins_by_group.png", bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "tab5_pod_bins.csv", index=False)


def fig5_2(eval_dir: Path, out: Path):
    pipes = ["A", "B", "C", "D", "J"]
    conds = list(COND)
    width = 0.15
    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=200)
    rows = []
    for i, cond in enumerate(conds):
        label, color, marker = COND[cond]
        ms = [metrics(eval_dir, r) for r in RUNS[cond]]
        means, seeds = [], []
        for p in pipes:
            v = [m[f"pipe_{p}"]["MTL"]["recall"] for m in ms]
            means.append(np.mean(v)); seeds.append(v)
            rows.append({"pipe": p, "cond": cond, "recall_s0": v[0], "recall_s1": v[1], "mean": np.mean(v)})
        x = np.arange(len(pipes)) + (i - 2) * (width + 0.01)
        ax.bar(x, means, width=width, color=color, label=label, linewidth=0)
        for xi, sv in zip(x, seeds):
            ax.plot([xi, xi], [min(sv), max(sv)], color=INK, lw=0.8, solid_capstyle="round")
            ax.scatter([xi] * 2, sv, s=6, color=INK, zorder=3)
    ax.set_xticks(range(len(pipes))); ax.set_xticklabels([PIPE_LABEL[p] for p in pipes])
    ax.set_ylabel("MTL 재현율 (신뢰도 0.25)"); ax.set_ylim(0, 0.75); ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.set_xlabel("시험 배관 (공칭 외경 mm)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5, frameon=False, handlelength=1.2, columnspacing=1.0)
    ax.text(0.99, 0.97, "막대: 두 시드 평균, 점: 시드 0·1", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "fig5-2_recall_by_pipe.png", bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "tab5_recall_by_pipe.csv", index=False)


def pooled_457(m: dict):
    tp = sum(m[f"pipe_{p}"]["MTL"]["recall"] * m[f"pipe_{p}"]["MTL"]["n_gt"] for p in "ABC")
    n = sum(m[f"pipe_{p}"]["MTL"]["n_gt"] for p in "ABC")
    fp = sum(m[f"pipe_{p}"]["MTL"]["fp"] for p in "ABC")
    return tp / n, tp / (tp + fp)


def fig5_3(eval_dir: Path, out: Path):
    sweep = [(0.749, None), (1.0, "ry100"), (1.165, "ry117"), (1.333, "ry133"), (1.6, "ry160")]
    xs = list(range(len(sweep)))                     # 설정값은 이산적이므로 등간격 배치
    xlabels = ["0.749\n(C 본래)", "1.0", "1.165\n(B 배율)", "1.333", "1.6"]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), dpi=200, gridspec_kw={"wspace": 0.3})
    for s, ls in ((0, "--"), (1, "-")):
        rec, prec = [], []
        for ry, suf in sweep:
            run = f"C_s{s}" if suf is None else f"C_s{s}_{suf}"
            m = metrics(eval_dir, run)
            r, p = pooled_457(m) if suf is None else (m["overall"]["MTL"]["recall"], m["overall"]["MTL"]["precision"])
            rec.append(r); prec.append(p); rows.append({"seed": s, "r_y": ry, "recall": r, "precision": p})
        c = COND["C"][1]
        axes[0].plot(xs, rec, color=c, ls=ls, marker="^", ms=4, lw=1.6, mec="white", mew=0.6, label=f"C 모델, 시드 {s}")
        axes[1].plot(xs, prec, color=c, ls=ls, marker="^", ms=4, lw=1.6, mec="white", mew=0.6, label=f"C 모델, 시드 {s}")
        mb = metrics(eval_dir, f"B_s{s}"); rb, pb = pooled_457(mb)
        axes[0].axhline(rb, color=COND["B"][1], ls=ls, lw=1.2, label=f"B 모델, 시드 {s} (기준)")
        axes[1].axhline(pb, color=COND["B"][1], ls=ls, lw=1.2, label=f"B 모델, 시드 {s} (기준)")
    for ax, ylab, title in zip(axes, ["MTL 재현율", "MTL 정밀도"], ["(a) 재현율", "(b) 정밀도"]):
        ax.set_xlabel("추론 세로 배율 r_y (가로 1.252 고정)"); ax.set_ylabel(ylab); ax.set_title(title, loc="left")
        ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8); ax.set_xlim(-0.3, len(xs) - 0.7)
        ax.grid(axis="y"); ax.set_axisbelow(True)
    axes[0].set_ylim(0.2, 0.45); axes[1].set_ylim(0.1, 0.3)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.15, -0.3), ncol=4, frameon=False, handlelength=2.0, columnspacing=1.2)
    fig.savefig(out / "fig5-3_scale_sweep.png", bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "tab5_scale_sweep.csv", index=False)


def fig5_4(defer_dir: Path, out: Path):
    runs = [("D (B 모델)", "D_sub", COND["B"][1]), ("E (C 모델)", "E_sub", COND["C"][1]),
            ("E− (C− 모델)", "Ec_sub", COND["C⁻"][1]), ("E′ (C′ 모델)", "Ep_sub", COND["C′"][1])]
    fig, ax = plt.subplots(figsize=(5.2, 3.3), dpi=200)
    rows = []
    for label, run, color in runs:
        f = defer_dir / run / "deferral.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        res = [v for k, v in d.items() if k.endswith("lambda0")][0]
        rc = res["risk_coverage"]
        xs = [r["review_pct"] for r in rc]; ys = [r["residual_miss_rate"] for r in rc]
        base = res["baseline_miss_rate"]
        ax.plot(xs, ys, color=color, lw=1.8, label=label)
        ax.plot(xs, [base * (1 - x / 100) for x in xs], color=color, lw=0.9, ls=":", alpha=0.9)
        for r in rc:
            rows.append({"cond": label, "review_pct": r["review_pct"], "residual_miss_rate": r["residual_miss_rate"],
                         "random_expected": base * (1 - r["review_pct"] / 100)})
    for k in (5, 10, 20):
        ax.axvline(k, color=GRAY, lw=0.6, ls="-", alpha=0.5)
    ax.set_xlim(0, 55); ax.set_ylim(0, 0.8)
    ax.set_xlabel("검사원 검토 비율 (영상 수 기준, %)"); ax.set_ylabel("잔여 미탐률 (손상 4종)")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([0], [0], color=INK2, lw=1.8), Line2D([0], [0], color=INK2, lw=0.9, ls=":")]
    l += ["실선: sub_thr_max 점수", "점선: 무작위 보류 기대값"]
    ax.legend(h, l, loc="upper right", frameon=False, ncol=2, handlelength=1.8, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig5-4_risk_coverage.png", bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "tab5_risk_coverage.csv", index=False)


def fig5_5(eval_dir: Path, out: Path):
    ns = [0, 50, 200]
    series = {"B": ("B_s1", "B_s1_ad50", "B_s1_ad200"), "C": ("C_s1", "C_s1_ad50", "C_s1_ad200")}
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), dpi=200)
    for cond, runs in series.items():
        label, color, marker = COND[cond]
        ms = [metrics(eval_dir, r) for r in runs]
        rec = [m["overall"]["MTL"]["recall"] for m in ms]
        ap = [m["overall"]["MTL"]["ap50"] for m in ms]
        swa = [m["overall"]["SWA"]["recall"] for m in ms]
        for ax, ys in zip(axes, (rec, ap, swa)):
            ax.plot(ns, ys, color=color, marker=marker, ms=4, lw=1.6, mec="white", mew=0.6, label=f"{cond} 모델 (시드 1)")
        for n, r, a, s in zip(ns, rec, ap, swa):
            rows.append({"cond": cond, "n_adapt": n, "mtl_recall": r, "mtl_ap50": a, "swa_recall": s})
    for ax, ylab, title in zip(axes, ["MTL 재현율", "MTL AP50 (신뢰도 ≥ 0.25)", "SWA 재현율"], ["(a)", "(b)", "(c)"]):
        ax.set_xticks(ns); ax.set_xlabel("배관당 적응 영상 수"); ax.set_ylabel(ylab); ax.set_title(title, loc="left")
        ax.grid(axis="y"); ax.set_axisbelow(True)
    axes[0].set_ylim(0.3, 0.6); axes[1].set_ylim(0.1, 0.3); axes[2].set_ylim(0.1, 0.5)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig5-5_adaptation.png", bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "tab5_adaptation.csv", index=False)


# ---------- v2 추가: 같은 정밀도 비교(PR 곡선), 판단보류 영상/박스 축, 검토–적응 루프 ----------
def fig5_6(sweep_test_csv, out: Path):
    """조건별 정밀도–재현율 곡선 (임계값 스윕), 전체와 457 mm 그룹. 두 시드는 실선/점선."""
    sw = pd.read_csv(sweep_test_csv); sw = sw[sw["cls"] == "MTL"]
    runs = {"A": ("A_s0", "A_s1"), "B": ("B_s0", "B_s1"), "C": ("C_s0", "C_s1"), "C⁻": ("C_s0_clamp", "C_s1_clamp"), "C′": ("Cp_s0", "Cp_s1")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=200, gridspec_kw={"wspace": 0.25})
    for ax, scope, title in zip(axes, ("overall", "G457"), ("(a) 시험 5개 배관 전체", "(b) 457 mm 배관 A·B·C")):
        for cond, (r0, r1) in runs.items():
            label, color, marker = COND[cond]
            for run, ls in ((r0, "--"), (r1, "-")):
                d = sw[(sw["name"] == run) & (sw["scope"] == scope)].sort_values("thr")
                d = d[d["recall"] >= 0.10]           # 표본이 적은 꼬리(높은 임계값) 제외
                ax.plot(d["precision"], d["recall"], color=color, ls=ls, lw=1.5, marker=marker, ms=3, mec="white", mew=0.5,
                        label=label if ls == "-" else None)
        ax.set_xlabel("MTL 정밀도"); ax.set_ylabel("MTL 재현율"); ax.set_title(title, loc="left")
        ax.set_xlim(0.05, 0.6); ax.set_ylim(0, 0.75); ax.grid(axis="both"); ax.set_axisbelow(True)
    axes[0].text(0.98, 0.97, "실선: 시드 1, 점선: 시드 0", transform=axes[0].transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.12, -0.25), ncol=5, frameon=False, handlelength=1.6, columnspacing=1.0)
    fig.savefig(out / "fig5-6_pr_curves.png", bbox_inches="tight"); plt.close(fig)


def _scores_df(defer_dir: Path, run: str):
    f = defer_dir / run / "image_scores_lambda0.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    def rank_pct(x):
        order = np.argsort(np.argsort(x.values, kind="stable"), kind="stable"); return pd.Series((order + 1) / max(1, len(x)), index=x.index)
    df["n_pred"] = df["n_pred_thr"].astype(float)
    df["combo"] = df.groupby("pipe")["n_pred_thr"].transform(rank_pct) + df.groupby("pipe")["sub_thr_max"].transform(rank_pct)
    return df


def fig5_7(defer_dir: Path, out: Path, run="E_sub", cond_label="E (C 모델)"):
    """판단보류 점수 3종의 Risk–Coverage: (a) 검토 영상 비율 축, (b) 검토 박스 수 축."""
    df = _scores_df(defer_dir, run)
    if df is None:
        raise FileNotFoundError(run)
    total_gt = df.n_gt.sum(); total_box = df.n_pred_thr.sum()
    scores = {"sub_thr_max (약한 검출)": ("sub_thr_max", COND["C"][1], "-"), "n_pred (검출 수)": ("n_pred", COND["B"][1], "-"),
              "combo (순위 합)": ("combo", COND["C′"][1], "-")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=200, gridspec_kw={"wspace": 0.28})
    for label, (col, color, ls) in scores.items():
        d = df.sort_values(col, ascending=False, kind="stable")
        miss_cum = np.concatenate([[0], np.cumsum(d.n_miss.values)]); box_cum = np.concatenate([[0], np.cumsum(d.n_pred_thr.values)])
        resid = (df.n_miss.sum() - miss_cum) / max(total_gt, 1)
        n = np.arange(len(d) + 1)
        axes[0].plot(100 * n / len(d), resid, color=color, lw=1.6, label=label)
        axes[1].plot(100 * box_cum / max(total_box, 1), resid, color=color, lw=1.6, label=label)
    base = df.n_miss.sum() / max(total_gt, 1)
    xs = np.linspace(0, 100, 51)
    axes[0].plot(xs, base * (1 - xs / 100), color=GRAY, ls=":", lw=1.0, label="무작위 보류 기대값")
    axes[1].plot(xs, base * (1 - xs / 100), color=GRAY, ls=":", lw=1.0)
    for ax, xl, title in zip(axes, ["검토 영상 비율 (%)", "검토 박스 비율 (운영 임계값 이상 검출 수, %)"], ["(a) 영상 수 기준", "(b) 박스 수 기준"]):
        ax.set_xlim(0, 50); ax.set_ylim(0, 0.8); ax.set_xlabel(xl); ax.set_ylabel("잔여 미탐률 (손상 4종)"); ax.set_title(title, loc="left")
        ax.grid(axis="y"); ax.set_axisbelow(True)
    axes[0].text(0.98, 0.97, cond_label, transform=axes[0].transAxes, ha="right", va="top", fontsize=8, color=INK2)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.12, -0.25), ncol=4, frameon=False, handlelength=1.6, columnspacing=1.0)
    fig.savefig(out / "fig5-7_risk_coverage_scores.png", bbox_inches="tight"); plt.close(fig)


def fig5_8(eval_sel_dir: Path, out: Path):
    """검토–적응 루프: 기준 / 무작위 200장 적응 / 선택 200장 적응 (MTL 재현율·정밀도·AP50), 모델별."""
    models = [m for m in ("A_s1", "B_s1", "C_s1") if (eval_sel_dir / m / "metrics.json").exists()]
    if not models:
        raise FileNotFoundError("eval_sel")
    conds = [("기준", ""), ("무작위 200장", "_ad200"), ("검토 대상 200장", "_adsel200")]
    colors = [GRAY, COND["B"][1], COND["C′"][1]]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), dpi=200, gridspec_kw={"wspace": 0.35})
    width = 0.25
    for ax, key, ylab in zip(axes, ("recall", "precision", "ap50"), ("MTL 재현율", "MTL 정밀도", "MTL AP50 (신뢰도 ≥ 0.25)")):
        for j, (clab, suf) in enumerate(conds):
            vals = []
            for m in models:
                f = eval_sel_dir / (m + suf) / "metrics.json"
                vals.append(json.loads(f.read_text(encoding="utf-8"))["overall"]["MTL"][key] if f.exists() else np.nan)
            x = np.arange(len(models)) + (j - 1) * (width + 0.02)
            ax.bar(x, vals, width=width, color=colors[j], label=clab, linewidth=0)
            for xi, v in zip(x, vals):
                if np.isfinite(v):
                    ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(range(len(models))); ax.set_xticklabels([m.replace("_s1", " 모델") for m in models]); ax.set_ylabel(ylab)
        ax.grid(axis="y"); ax.set_axisbelow(True); ax.set_ylim(0, 0.6)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    fig.savefig(out / "fig5-8_review_adapt_loop.png", bbox_inches="tight"); plt.close(fig)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval"); ap.add_argument("--defer", default="defer")
    ap.add_argument("--out", default="analysis/ch5")
    ap.add_argument("--sweep-test", default="analysis/sweep_test.csv")
    ap.add_argument("--eval-sel", default="eval_sel")
    ap.add_argument("--defer-run", default="E_sub", help="그림 5-7에 쓸 defer 폴더 (예: DA_sub)")
    ap.add_argument("--defer-label", default="E (C 모델)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ev, df = Path(a.eval), Path(a.defer)
    for fn, args in ((fig5_6, (a.sweep_test, out)), (lambda d, o: fig5_7(d, o, run=a.defer_run, cond_label=a.defer_label), (df, out)), (fig5_8, (Path(a.eval_sel), out))):
        try:
            fn(*args); print("완료:", fn.__name__)
        except Exception as e:
            print(f"건너뜀 {fn.__name__}: {e}")
    for fn in (fig5_1, fig5_2, fig5_3, fig5_5):
        try:
            fn(ev, out); print("완료:", fn.__name__)
        except Exception as e:
            print(f"건너뜀 {fn.__name__}: {e}")
    try:
        fig5_4(df, out); print("완료: fig5_4")
    except Exception as e:
        print(f"건너뜀 fig5_4: {e}")
    print("그림 저장:", out.resolve())


if __name__ == "__main__":
    main()
