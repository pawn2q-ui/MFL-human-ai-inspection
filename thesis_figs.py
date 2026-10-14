"""
논문용 그림 3-1, 3-2 생성 (analyze_dataset 의 all_boxes.csv 로부터).
  python thesis_figs.py --boxes analysis/all_boxes.csv --out analysis
그림 3-1: 금속 손실(MTL) 박스의 원주 방향 길이 분포 — 픽셀 기준 vs mm 기준, 관경별
그림 3-2: 배관별 MTL 박스 가로세로비(세로/가로) 중앙값 — 픽셀 기준 vs mm 기준, 1/D 예측선
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 글꼴: 있는 것 중 첫 번째
for name in ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.linewidth": 0.8, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "grid.color": "#e6e5e1", "grid.linewidth": 0.6})

RAMP = {457: "#86b6ef", 508: "#5598e7", 610: "#2a78d6", 711: "#1c5cab", 813: "#0d366b"}   # 관경 순서(밝→어둠)
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
IMG_H = 2400


def fig3_1(boxes: pd.DataFrame, out: Path, cls="MTL"):
    d = boxes[boxes["cls"] == cls]
    pipes_by_D = d.groupby("D")["pipe"].agg(lambda s: ",".join(sorted(set(s))))
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.4), dpi=200)
    edges = np.logspace(1, 3.2, 34)   # 10 ~ 1585 (로그 간격)
    for ax, col, xlabel, title in [(axes[0], "h_px", "박스 세로 길이 (픽셀)", "(a) 픽셀 기준"),
                                   (axes[1], "h_mm", "박스 세로 길이, 원주 방향 (mm)", "(b) mm 기준 (관경 환산)")]:
        for D, g in d.groupby("D"):
            v = g[col].values
            v = v[v > 0]
            dens, _ = np.histogram(v, bins=edges, density=True)
            centers = np.sqrt(edges[:-1] * edges[1:])
            ax.plot(centers, dens * centers, color=RAMP[int(D)], lw=1.6,
                    label=f"{int(D)} mm ({pipes_by_D[D]}, n={len(v):,})")
            med = np.median(v)
            ax.axvline(med, color=RAMP[int(D)], lw=0.8, ls=":", alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlim(10, 1500)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.grid(True, axis="y")
        ax.set_yticks([])
        ax.set_ylabel("상대 빈도" if ax is axes[0] else "")
    # 중앙값 표기: 관경 최소·최대만 직접 라벨 (왼쪽 위 텍스트 블록)
    for ax, col in [(axes[0], "h_px"), (axes[1], "h_mm")]:
        lines = []
        for D in (457, 813):
            med = np.median(d[d["D"] == D][col])
            lines.append((D, f"{int(D)} mm 중앙값 {med:.0f}"))
        for i, (D, txt) in enumerate(lines):
            ax.text(0.02, 0.97 - 0.09 * i, txt, color=RAMP[D], fontsize=7.5, ha="left", va="top", transform=ax.transAxes)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, title="공칭 외경 (배관, MTL 박스 수)",
               title_fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out / "fig3-1_MTL_height_px_vs_mm.png", dpi=200)
    plt.close(fig)


def fig3_2(boxes: pd.DataFrame, out: Path, cls="MTL"):
    d = boxes[boxes["cls"] == cls]
    t = d.groupby(["pipe", "D"]).agg(ar_px=("ar_px", "median"), ar_mm=("ar_mm", "median"), n=("ar_px", "size")).reset_index()
    t = t.sort_values(["D", "pipe"])
    # 같은 관경의 배관은 x 를 조금씩 벌려서 표시
    xs = []
    for D, g in t.groupby("D"):
        k = len(g)
        offs = (np.arange(k) - (k - 1) / 2) * 9
        xs += list(D + offs)
    t["x"] = xs
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    # 1/D 예측선: 물리적 형상이 일정하다면 픽셀 가로세로비 ∝ 2400/(πD). 전체 mm 중앙값으로 고정
    c = float(d["ar_mm"].median())
    Dgrid = np.linspace(430, 840, 100)
    ax.plot(Dgrid, c * IMG_H / (math.pi * Dgrid), color=GRAY, lw=1.2, ls="--",
            label=f"물리 형상이 일정할 때의 픽셀 기준 예측 (∝ 1/D, mm 중앙값 {c:.2f} 기준)")
    ax.plot(t["x"], t["ar_px"], "o", color=BLUE, ms=6, label="픽셀 기준 중앙값 (배관별)")
    ax.plot(t["x"], t["ar_mm"], "s", color=ORANGE, ms=5.5, label="mm 기준 중앙값 (배관별)")
    for _, r in t.iterrows():
        dx = -7 if r["D"] >= 800 else 0
        ax.text(r["x"] + dx, r["ar_px"] + 0.05, r["pipe"], color=BLUE, fontsize=7, ha="center" if dx == 0 else "right", va="bottom")
    # 관경별 중앙값 추세(가는 선)
    m = d.groupby("D").agg(ar_px=("ar_px", "median"), ar_mm=("ar_mm", "median")).reset_index()
    ax.plot(m["D"], m["ar_px"], color=BLUE, lw=0.9, alpha=0.6)
    ax.plot(m["D"], m["ar_mm"], color=ORANGE, lw=0.9, alpha=0.6)
    ax.set_xlabel("공칭 외경 D (mm)")
    ax.set_ylabel("MTL 박스 가로세로비 (세로 / 가로)")
    ax.set_xticks([457, 508, 610, 711, 813])
    ax.set_ylim(0.8, 3.0)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig3-2_MTL_aspect_ratio_vs_D.png", dpi=200)
    plt.close(fig)
    return t, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", default="analysis/all_boxes.csv")
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    boxes = pd.read_csv(a.boxes)
    fig3_1(boxes, out)
    t, m = fig3_2(boxes, out)
    t.to_csv(out / "fig3-2_MTL_aspect_by_pipe.csv", index=False)
    print(m.round(2).to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main()
