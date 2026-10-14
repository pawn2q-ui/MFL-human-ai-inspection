"""
배관별로 다른 예측 폴더를 하나로 묶는다 (추론 배율 실험 결과를 5개 배관 조건으로 합칠 때).

예: '축소 금지' 추론 조건(C⁻) = 457 mm 배관(A,B,C)은 r_y=1.0 결과, 508 mm(D)도 r_y=1.0 결과, 813 mm(J)는 원래 C 결과
  python -m mfl_thesis.merge_preds --manifest manifest.csv --split test --out preds/C_s0_clamp ^
      --src preds/C_s0_ry100:A,B,C --src preds/C_s0_D100:D --src preds/C_s0:J

출력은 infer_images 와 같은 형식 (<out>/preds/<stem>.txt, tile_stats.json, DONE) 이므로 evaluate / deferral 이 그대로 읽는다.
해당 split 의 영상이 어느 --src 에도 없으면 오류로 멈춘다 (빠진 배관이 없도록).
"""
import argparse
import json
import shutil
from pathlib import Path

from .manifest import read_manifest


def merge(manifest, splits, out, sources, verbose=True):
    rows = [r for r in read_manifest(manifest) if r["split"] in splits]
    pipe_src = {}
    for src, pipes in sources:
        for p in pipes:
            if p in pipe_src:
                raise ValueError(f"배관 {p} 가 두 번 지정됨: {pipe_src[p]} / {src}")
            pipe_src[p] = Path(src)
    out = Path(out)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    images = {}
    missing = []
    n = 0
    for r in rows:
        src = pipe_src.get(r["pipe_id"])
        if src is None:
            missing.append(f"{r['stem']} (배관 {r['pipe_id']} 의 --src 없음)"); continue
        f = src / "preds" / f"{r['stem']}.txt"
        if not f.exists():
            missing.append(str(f)); continue
        shutil.copy2(f, out / "preds" / f"{r['stem']}.txt")
        n += 1
    if missing:
        raise FileNotFoundError(f"예측 파일 {len(missing)}개가 없습니다. 예: {missing[:3]}")
    for src in set(pipe_src.values()):
        f = src / "tile_stats.json"
        if f.exists():
            st = json.loads(f.read_text(encoding="utf-8")).get("images", {})
            for stem, v in st.items():
                if pipe_src.get(v.get("pipe")) == src and v.get("split") in splits:
                    images[stem] = v
    (out / "tile_stats.json").write_text(json.dumps({"cond": "merged", "sources": {p: str(s) for p, s in pipe_src.items()},
                                                     "images": images}, ensure_ascii=False))
    (out / "DONE").write_text("ok")
    if verbose:
        print(f"묶음 완료: {n} 영상 → {out}  ({', '.join(f'{p}←{s}' for p, s in sorted(pipe_src.items()))})")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", action="append", required=True, help="'예측폴더:배관,배관' (여러 번 지정)")
    a = ap.parse_args()
    sources = []
    for s in a.src:
        folder, pipes = s.rsplit(":", 1)
        sources.append((folder, [p.strip() for p in pipes.split(",") if p.strip()]))
    merge(a.manifest, set(a.split.split(",")), a.out, sources)


if __name__ == "__main__":
    main()
