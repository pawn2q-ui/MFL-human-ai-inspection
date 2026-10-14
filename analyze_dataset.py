"""
2주차 데이터 분석 (논문 3.2절 재료)
  표 3-1  : 배관별 관경, s_y(mm/px), 영상 수, 양성 영상 수, 손상 클래스별 박스 수
  그림 3-1: 손상 박스의 세로 길이 분포 — 픽셀 vs mm (배관별)
  그림 3-2: 손상 박스의 가로세로비 — 픽셀 기준 vs mm 기준 (배관별)
  기타    : 영상당 박스 수, 빈 영상 비율, 원주 이음부(위/아래 경계)에 걸친 박스 수

사용: python -m mfl_thesis.analyze_dataset --manifest manifest.csv --out analysis/
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import IMG_W, IMG_H, CLASSES, DAMAGE, DAMAGE_IDS, s_y
from .labels import read_yolo
from .manifest import read_manifest

SEAM_PX = 2   # 위/아래 경계에서 이 픽셀 이내면 '이음부에 걸침'으로 간주


def collect_boxes(manifest_rows):
    rows, img_rows = [], []
    for r in manifest_rows:
        boxes = read_yolo(r["label"], IMG_W, IMG_H) if r["label"] else np.zeros((0, 5))
        sy = s_y(r["diameter_mm"])
        img_rows.append({"stem": r["stem"], "pipe": r["pipe_id"], "split": r["split"], "n_box": len(boxes),
                         "n_damage": int(np.isin(boxes[:, 0], DAMAGE_IDS).sum()) if len(boxes) else 0})
        for c, x1, y1, x2, y2 in boxes:
            w, h = x2 - x1, y2 - y1
            rows.append({"stem": r["stem"], "pipe": r["pipe_id"], "split": r["split"], "D": r["diameter_mm"],
                         "cls": CLASSES[int(c)], "w_px": w, "h_px": h, "w_mm": w * 1.0, "h_mm": h * sy,
                         "ar_px": h / max(w, 1e-6), "ar_mm": (h * sy) / max(w, 1e-6),
                         "seam": bool(y1 <= SEAM_PX or y2 >= IMG_H - SEAM_PX)})
    return pd.DataFrame(rows), pd.DataFrame(img_rows)


def table_3_1(boxes: pd.DataFrame, imgs: pd.DataFrame) -> pd.DataFrame:
    pipes = imgs.groupby("pipe").agg(n_img=("stem", "size"), n_pos=("n_box", lambda s: int((s > 0).sum())),
                                     split=("split", lambda s: "/".join(sorted(set(s)))))
    D = boxes.groupby("pipe")["D"].first() if len(boxes) else pd.Series(dtype=float)
    pipes["D_mm"] = D
    pipes["s_y_mm_per_px"] = pipes["D_mm"].map(lambda d: s_y(d) if pd.notna(d) else np.nan)
    for c in DAMAGE:
        pipes[f"n_{c}"] = boxes[boxes.cls == c].groupby("pipe").size()
    pipes["n_box_total"] = boxes.groupby("pipe").size()
    pipes = pipes.fillna(0)
    cols = ["split", "D_mm", "s_y_mm_per_px", "n_img", "n_pos"] + [f"n_{c}" for c in DAMAGE] + ["n_box_total"]
    return pipes[cols].sort_values("D_mm")


def fig_3_1(boxes: pd.DataFrame, out: Path):
    d = boxes[boxes.cls.isin(DAMAGE)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for pipe, g in d.groupby("pipe"):
        D = g["D"].iloc[0]
        axes[0].hist(g["h_px"], bins=40, histtype="step", label=f"{pipe} ({D:.0f} mm)")
        axes[1].hist(g["h_mm"], bins=40, histtype="step", label=f"{pipe} ({D:.0f} mm)")
    axes[0].set_xlabel("box height (px)"); axes[1].set_xlabel("box height, circumferential (mm)")
    axes[0].set_ylabel("count"); axes[0].legend(fontsize=7); axes[0].set_title("pixel scale"); axes[1].set_title("physical scale")
    fig.tight_layout(); fig.savefig(out / "fig3-1_box_height_px_vs_mm.png", dpi=150); plt.close(fig)


def fig_3_2(boxes: pd.DataFrame, out: Path):
    d = boxes[boxes.cls.isin(DAMAGE)]
    g = d.groupby("pipe").agg(D=("D", "first"), ar_px=("ar_px", "median"), ar_mm=("ar_mm", "median")).sort_values("D")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g["D"], g["ar_px"], "o-", label="median h/w (pixels)")
    ax.plot(g["D"], g["ar_mm"], "s--", label="median h/w (mm)")
    ax.set_xlabel("nominal outer diameter (mm)"); ax.set_ylabel("aspect ratio (height / width)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig3-2_aspect_ratio_px_vs_mm.png", dpi=150); plt.close(fig)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(a.manifest)
    boxes, imgs = collect_boxes(rows)
    t = table_3_1(boxes, imgs)
    t.to_csv(out / "table3-1_pipes.csv")
    print("표 3-1\n", t.round(3).to_string())
    fig_3_1(boxes, out)
    ar = fig_3_2(boxes, out)
    ar.to_csv(out / "fig3-2_aspect_ratio.csv")
    # 기타 통계
    extra = {
        "n_images": int(len(imgs)),
        "empty_ratio": float((imgs.n_box == 0).mean()),
        "boxes_per_positive_image": float(imgs.loc[imgs.n_box > 0, "n_box"].mean()) if (imgs.n_box > 0).any() else 0.0,
        "damage_boxes": int(boxes.cls.isin(DAMAGE).sum()),
        "seam_touching_damage_boxes": int((boxes.cls.isin(DAMAGE) & boxes.seam).sum()),
        "class_counts": boxes.cls.value_counts().to_dict(),
    }
    pd.Series(extra).to_json(out / "extra_stats.json", force_ascii=False, indent=2)
    boxes.to_csv(out / "all_boxes.csv", index=False)
    print("기타:", extra)
    # 논문용 그림(관경별 MTL 분포, 가로세로비 vs D)
    try:
        from .thesis_figs import fig3_1 as tf1, fig3_2 as tf2
        tf1(boxes, out)
        tf2(boxes, out)
    except Exception as e:   # 글꼴 등 환경 문제로 실패해도 본 분석은 유지
        print("논문용 그림 생성 실패:", e)
    print(f"-> {out}/ 에 저장")


if __name__ == "__main__":
    main()
