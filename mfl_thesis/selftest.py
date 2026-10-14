"""
자체 검증: 합성 데이터로 전체 파이프라인을 끝까지 돌려 본다.
  합성 생성 -> 매니페스트 -> 데이터 분석 -> 타일 캐시(A/B/C) -> (완벽/모형) 추론 -> 평가(POD·a90) -> 부트스트랩 -> 판단보류

사용: python -m mfl_thesis.selftest [--work /tmp/mfl_selftest] [--n-per-pipe 8]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from . import synth_data, make_tiles, infer_tiles, evaluate, deferral
from .config import IMG_W, IMG_H, DAMAGE_IDS
from .metadata import PipeResolver, load_pipes
from .manifest import build_manifest, write_manifest, read_manifest
from .labels import read_yolo, read_pred, iou_matrix


def run_cli(mod, *args):
    cmd = [sys.executable, "-m", f"mfl_thesis.{mod}", *map(str, args)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    if r.returncode != 0:
        print(r.stdout); print(r.stderr); raise SystemExit(f"실패: {' '.join(cmd)}")
    return r.stdout


def roundtrip_check(manifest, tiles_dir, preds_dir):
    """완벽 예측기(라벨 그대로)로 추론했을 때, 원본 좌표로 되돌린 예측이 정답과 일치하는지."""
    rows = [r for r in read_manifest(manifest) if r["split"] == "test"]
    ious, n_gt, n_hit = [], 0, 0
    for r in rows:
        gt = read_yolo(r["label"], IMG_W, IMG_H)
        pr = read_pred(Path(preds_dir) / f"{r['stem']}.txt")
        if len(gt) == 0:
            continue
        m = iou_matrix(gt[:, 1:5], pr[:, 2:6]) if len(pr) else np.zeros((len(gt), 0))
        best = m.max(1) if m.size else np.zeros(len(gt))
        n_gt += len(gt); n_hit += int((best >= 0.5).sum()); ious += list(best)
    return n_hit / max(n_gt, 1), float(np.mean(ious)) if ious else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/mfl_selftest")
    ap.add_argument("--n-per-pipe", type=int, default=8)
    ap.add_argument("--T", type=int, default=1280)
    a = ap.parse_args()
    work = Path(a.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    print("1) 합성 데이터 생성")
    data = synth_data.generate(work / "data", n_per_pipe=a.n_per_pipe, seed=0)
    pipes = load_pipes(data / "pipes.csv")
    rows = build_manifest(data / "images", data / "labels", pipes, PipeResolver("regex", r"^([A-E])_"),
                          test_pipes={"C", "E"}, val_pipe="D", adapt_n=2, seed=0)
    man = work / "manifest.csv"; write_manifest(rows, man)
    from collections import Counter
    print("   split:", dict(Counter((r[5], r[3]) for r in rows)))

    print("2) 데이터 분석 (표 3-1, 그림 3-1/3-2)")
    out = run_cli("analyze_dataset", "--manifest", man, "--out", work / "analysis")
    t = pd.read_csv(work / "analysis" / "table3-1_pipes.csv", index_col=0)
    assert len(t) == 5, t
    print(t[["D_mm", "s_y_mm_per_px", "n_img", "n_pos", "n_box_total"]].round(3).to_string())

    print("3) 타일 캐시 생성")
    info = {}
    for cond in ["A", "B", "C"]:
        info[(cond, "train")] = make_tiles.build(man, cond, {"train"}, "train", work / f"tiles_{cond}_train", T=a.T, verbose=False)
        info[(cond, "eval")] = make_tiles.build(man, cond, {"test", "val"}, "infer", work / f"tiles_{cond}_eval", T=a.T, verbose=False)
        print(f"   {cond}: train 타일 {info[(cond,'train')]['n_tiles']} (pos {info[(cond,'train')]['n_pos']}, neg {info[(cond,'train')]['n_neg']}), eval 타일 {info[(cond,'eval')]['n_tiles']}")
    make_tiles.build(man, "C", {"test", "val"}, "infer", work / "tiles_C_eval_nowrap", T=a.T, wrap=False, verbose=False)
    assert info[("C", "train")]["s_star"] > 0

    print("4) 기하 왕복 검증 (완벽 예측기)")
    for cond in ["A", "C"]:
        td = work / f"tiles_{cond}_eval"
        pf = synth_data.mock_predictor(td / "labels", info[(cond, "eval")]["T"], perfect=True)
        infer_tiles.run_inference(td, pf, work / f"preds_{cond}_perfect", verbose=False)
        rec, miou = roundtrip_check(man, td, work / f"preds_{cond}_perfect" / "preds")
        print(f"   {cond}: 정답 박스 recall(IoU>=0.5)={rec:.3f}, 평균 최대 IoU={miou:.3f}")
        assert rec > 0.97 and miou > 0.9, "기하 왕복 오류"

    print("5) 모형 추론 (시드 2개) + 평가")
    gts = {}
    for cond in ["A", "B", "C"]:
        td = work / f"tiles_{cond}_eval"
        for seed in [0, 1]:
            pf = synth_data.mock_predictor(td / "labels", info[(cond, "eval")]["T"], seed=seed)
            infer_tiles.run_inference(td, pf, work / f"preds_{cond}_s{seed}", verbose=False)
            gt, pr, metrics, pod, bins = evaluate.evaluate(man, work / f"preds_{cond}_s{seed}" / "preds", {"test"},
                                                           work / f"eval_{cond}_s{seed}", n_boot=50, verbose=False)
            gts[(cond, seed)] = gt
            o = metrics["overall"]["DMG_pooled"]; pa = pod[(pod.cls == "DMG_pooled")]
            print(f"   {cond} s{seed}: recall={o['recall']:.3f} miss={o['miss_rate']:.3f} mAP50={o['mAP50']:.3f} | a90(mm) " +
                  ", ".join(f"{r.pipe}:{r.a90:.0f}" if np.isfinite(r.a90) else f"{r.pipe}:nan" for r in pa.itertuples()))
    bs = evaluate.paired_bootstrap(gts[("A", 0)], gts[("C", 0)], n=300)
    print(f"   A−C 미탐률 차이 = {bs['miss_diff_a_minus_b']:+.3f}  95% CI [{bs['ci_lo']:+.3f}, {bs['ci_hi']:+.3f}] (박스 {bs['n_boxes']})")

    print("6) 판단보류 (조건 C 모델, 시험 배관)")
    rows_all = read_manifest(man); test_rows = [r for r in rows_all if r["split"] == "test"]
    s_min, s_max = deferral.train_s_range(rows_all)
    df = deferral.image_table(test_rows, work / "preds_C_s0" / "preds", work / "preds_C_s1" / "preds")
    results = {}
    for score in ["sub_thr_max", "one_minus_max", "ensemble", "random"]:
        for lam in ([0.0, 1.0] if score == "sub_thr_max" else [0.0]):
            res, _ = deferral.evaluate_deferral(df, score, lam, s_min, s_max, ks=(10, 20), n_random=50)
            results[f"{score}_l{lam:g}"] = res
            k20 = res["k"]["20"]
            print(f"   {score:14s} λ={lam:g}: 기준 미탐 {res['baseline_miss_rate']:.3f} | 20% 보류 → 잔여 {k20['score_based']['residual_miss_rate']:.3f} "
                  f"(무작위 {k20['random_mean_std']['residual_miss_rate'][0]:.3f}±{k20['random_mean_std']['residual_miss_rate'][1]:.3f}) "
                  f"포착 {k20['score_based']['capture_rate']:.2f} AUROC {res.get('auroc_has_miss', float('nan')):.2f}")
    deferral.plot_risk_coverage(results, work / "risk_coverage.png")
    (work / "deferral.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"\n자체 검증 통과. 산출물: {work}")


if __name__ == "__main__":
    main()
