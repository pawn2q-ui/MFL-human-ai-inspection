"""
검토 대상 선택 (루프 실험 H6): 모델 예측만으로(라벨 미사용) 시험 배관별 검토 우선순위 상위 n장을 고르고,
그 영상의 split 을 'adapt_sel' 로 바꾼 새 매니페스트를 쓴다. 나머지 test 영상이 공통 평가 집합이 된다.

  python -m mfl_thesis.select_review --manifest manifest.csv --preds preds/B_s1/preds --split test --n 200 ^
      --score combo --out manifest_selB.csv --list analysis/sel_B.txt

점수
  n_pred   : 운영 임계값(0.25) 이상 검출 수
  sub      : 임계값 미만 검출의 최대 신뢰도 (sub_thr_max)
  combo    : 배관 안에서 n_pred 순위 백분율 + sub 순위 백분율 (기본)
보고용으로 선택 집합과 기존 무작위 적응 집합(split=adapt)의 정답 박스 수를 라벨에서 세어 출력한다(선택에는 쓰지 않음).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from .config import A_CONF, IMG_W, IMG_H, DAMAGE_IDS
from .labels import read_pred, read_yolo
from .manifest import read_manifest


def image_scores(rows, preds_dir, thr=A_CONF):
    preds_dir = Path(preds_dir)
    out = []
    for r in rows:
        p = read_pred(preds_dir / f"{r['stem']}.txt")
        n_pred = int((p[:, 1] >= thr).sum()) if len(p) else 0
        sub = p[p[:, 1] < thr][:, 1]
        out.append({"stem": r["stem"], "pipe": r["pipe_id"], "n_pred": n_pred, "sub": float(sub.max()) if len(sub) else 0.0})
    return out


def rank_pct(vals):
    order = np.argsort(np.argsort(vals, kind="stable"), kind="stable")
    return (order + 1) / max(1, len(vals))


def select(manifest, preds_dir, split, n, score="combo", thr=A_CONF, out=None, list_path=None, verbose=True):
    rows = read_manifest(manifest)
    cand = [r for r in rows if r["split"] == split]
    sc = image_scores(cand, preds_dir, thr)
    chosen = set()
    pipes = sorted({s["pipe"] for s in sc})
    for pid in pipes:
        idx = [i for i, s in enumerate(sc) if s["pipe"] == pid]
        n_pred = np.array([sc[i]["n_pred"] for i in idx], float)
        sub = np.array([sc[i]["sub"] for i in idx], float)
        if score == "n_pred":
            val = n_pred + 1e-6 * sub
        elif score == "sub":
            val = sub
        else:
            val = rank_pct(n_pred) + rank_pct(sub)
        top = np.argsort(-val, kind="stable")[:n]
        for t in top:
            chosen.add(sc[idx[t]]["stem"])
    # 새 매니페스트
    out_rows = []
    for r in rows:
        rr = dict(r)
        if rr["stem"] in chosen:
            rr["split"] = "adapt_sel"
        out_rows.append(rr)
    if out:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stem", "image", "label", "pipe_id", "diameter_mm", "split"])
            for rr in out_rows:
                w.writerow([rr["stem"], rr["image"], rr["label"], rr["pipe_id"], rr["diameter_mm"], rr["split"]])
    if list_path:
        Path(list_path).parent.mkdir(parents=True, exist_ok=True)
        Path(list_path).write_text("\n".join(sorted(chosen)), encoding="utf-8")
    if verbose:
        def boxes(split_name):
            tot = {}
            for rr in out_rows:
                if rr["split"] != split_name:
                    continue
                gt = read_yolo(rr["label"], IMG_W, IMG_H) if rr["label"] else np.zeros((0, 5))
                nd = int(np.isin(gt[:, 0], DAMAGE_IDS).sum()) if len(gt) else 0
                tot[rr["pipe_id"]] = tot.get(rr["pipe_id"], 0) + nd
            return tot
        b_sel, b_rand = boxes("adapt_sel"), boxes("adapt")
        print(f"선택 {len(chosen)}장 (배관별 {n}장, 점수 {score}) → split=adapt_sel; 남은 {split} 영상 {sum(1 for rr in out_rows if rr['split'] == split)}장")
        print("손상 박스 수(보고용) 선택 집합:", b_sel, "합계", sum(b_sel.values()))
        print("손상 박스 수(보고용) 무작위 적응 집합:", b_rand, "합계", sum(b_rand.values()))
    return chosen, out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--preds", required=True, help="선택에 쓸 모델의 preds/ 폴더")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200, help="배관별 선택 영상 수")
    ap.add_argument("--score", default="combo", choices=["combo", "n_pred", "sub"])
    ap.add_argument("--thr", type=float, default=A_CONF)
    ap.add_argument("--out", required=True, help="새 매니페스트 경로")
    ap.add_argument("--list", default=None, help="선택된 stem 목록 저장 경로")
    a = ap.parse_args()
    select(a.manifest, a.preds, a.split, a.n, a.score, a.thr, a.out, a.list)


if __name__ == "__main__":
    main()
