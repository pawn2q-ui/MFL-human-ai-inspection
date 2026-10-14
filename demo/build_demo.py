"""데모 HTML 생성: 결과 폴더(defer/, analysis/sweep_test.csv, eval_sel/)에서 데이터를 뽑아 단일 HTML(오프라인 실행)로 묶는다.
  python3 build_demo.py --results <results_dir> --out demo.html
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--out", default="demo.html")
ap.add_argument("--template", default=str(Path(__file__).parent / "demo_template.html"))
a = ap.parse_args()
R = Path(a.results)

# ---------- 판단보류 곡선 ----------
RUNS = [("DA_sub", "A 공식 프로토콜 (조건 D_A)"), ("D_sub", "B 동일 배율 (조건 D)"), ("E_sub", "C 관경 보정 (조건 E)"),
        ("Ec_sub", "C⁻ 축소 금지 추론 (조건 E⁻)"), ("Ep_sub", "C′ 축소 없는 보정 (조건 E′)")]
PIPES = ["A", "B", "C", "D", "J"]


def rank_pct(x):
    order = np.argsort(np.argsort(x.values, kind="stable"), kind="stable")
    return pd.Series((order + 1) / max(1, len(x)), index=x.index)


defer = {}
for run, label in RUNS:
    f = R / "defer" / run / "image_scores_lambda0.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f)
    df["n_pred"] = df["n_pred_thr"].astype(float) + 1e-6 * df["sub_thr_max"]
    df["combo"] = df.groupby("pipe")["n_pred_thr"].transform(rank_pct) + df.groupby("pipe")["sub_thr_max"].transform(rank_pct)
    n = len(df); total_gt = int(df.n_gt.sum()); total_miss = int(df.n_miss.sum()); total_box = int(df.n_pred_thr.sum())
    pipe_n = df.groupby("pipe").size().reindex(PIPES).fillna(0).astype(int).to_dict()
    scores = {}
    for sc in ["combo", "n_pred", "sub_thr_max"]:
        d = df.sort_values(sc, ascending=False, kind="stable")
        miss_cum = np.cumsum(d.n_miss.values); box_cum = np.cumsum(d.n_pred_thr.values)
        ks = list(range(0, 51))
        resid, boxes, cap, per_pipe = [], [], [], {p: [] for p in PIPES}
        pipe_arr = d["pipe"].values
        for k in ks:
            m = int(round(n * k / 100))
            mc = miss_cum[m - 1] if m > 0 else 0; bc = box_cum[m - 1] if m > 0 else 0
            resid.append(round((total_miss - mc) / max(total_gt, 1), 4)); boxes.append(int(bc)); cap.append(round(mc / max(total_miss, 1), 4))
            sel = pipe_arr[:m]
            for p in PIPES:
                per_pipe[p].append(round(float((sel == p).sum() / max(pipe_n.get(p, 1), 1)), 3))
        scores[sc] = {"resid": resid, "boxes": boxes, "cap": cap, "per_pipe": per_pipe}
    defer[run] = {"label": label, "n": n, "total_gt": total_gt, "total_miss": total_miss, "total_box": total_box,
                  "base": round(total_miss / max(total_gt, 1), 4), "pipe_n": pipe_n, "scores": scores}

# ---------- 같은 정밀도 비교 (PR 곡선) ----------
pr = {}
f = R / "analysis" / "sweep_test.csv"
if f.exists():
    st = pd.read_csv(f); st = st[st.cls == "MTL"]
    for name in sorted(st.name.unique()):
        for scope in ["overall", "G457", "D", "J"]:
            d = st[(st.name == name) & (st.scope == scope)].sort_values("thr")
            d = d[(d.recall >= 0.10) & d.precision.notna()]     # 임계값이 높아 예측이 거의 없는 꼬리는 제외(그림 5-6과 동일)
            pr.setdefault(name, {})[scope] = {"thr": d.thr.round(2).tolist(), "p": d.precision.round(4).tolist(), "r": d.recall.round(4).tolist(), "fp": d.fp.astype(int).tolist(), "n_gt": int(d.n_gt.iloc[0]) if len(d) else 0}

# ---------- 루프 ----------
loop = {}
for model in ["A", "B", "C"]:
    rows = {}
    for tag, label in [("", "base"), ("_ad200", "random"), ("_adsel200", "selected")]:
        d = R / "eval_sel" / f"{model}_s1{tag}" / "metrics.json"
        if not d.exists():
            continue
        m = json.load(open(d))
        g = pd.read_csv(d.parent / "gt_records.csv"); mtl = g[g.cls == "MTL"]
        def rg(pipes):
            x = mtl[mtl["pipe"].isin(pipes)]; return round(float(x.hit.mean()), 4) if len(x) else None
        o = m["overall"]["MTL"]
        rows[label] = {"recall": round(o["recall"], 4), "precision": round(o["precision"], 4), "fp": int(o["fp"]), "ap50": round(o["ap50"], 4),
                       "r457": rg(["A", "B", "C"]), "rD": rg(["D"]), "rJ": rg(["J"]), "swa": round(m["overall"]["SWA"]["recall"], 4),
                       "dmg_map": round(m["overall"]["DMG_pooled"]["mAP50"], 4), "n_gt": int(o["n_gt"])}
    if rows:
        loop[model] = rows

data = {"defer": defer, "pr": pr, "loop": loop,
        "meta": {"n_test_images": 13740, "adapt_boxes": {"selected": "4,252–4,408", "random": 1412}, "adapt_pos": {"selected": "798–816", "random": 460}}}
html = Path(a.template).read_text(encoding="utf-8").replace("/*__DATA__*/", "const DATA = " + json.dumps(data, ensure_ascii=False) + ";")
Path(a.out).write_text(html, encoding="utf-8")
print("저장:", a.out, "| defer runs:", list(defer), "| pr:", list(pr), "| loop:", list(loop))
