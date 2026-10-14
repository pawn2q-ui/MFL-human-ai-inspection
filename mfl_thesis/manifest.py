"""
매니페스트(manifest.csv) 생성: 모든 후속 단계의 뼈대.
열: stem, image, label, pipe_id, diameter_mm, split   (split ∈ train / val / test / adapt / unused)

사용 예
  python -m mfl_thesis.manifest --images D:/pipemfl/test_data/images --labels D:/pipemfl/test_data/labels \
      --pipes pipes.csv --resolver regex --pattern "^([A-Z])[_-]" \
      --test-pipes K,E --val-pipe C --adapt-n 200 --out manifest.csv

split 규칙(3.3절): 시험 배관 2개(그중 하나는 관경이 학습 범위 끝), 검증 1개, 나머지 학습.
각 시험 배관에서 adapt-n 장을 '적응용'으로 분리(모든 평가에서 제외).
"""
import argparse
import csv
import random
from pathlib import Path

from .metadata import PipeResolver, load_pipes

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_manifest(images_dir, labels_dir, pipes, resolver, test_pipes, val_pipe, adapt_n=0, seed=0, max_per_pipe=None):
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    rng = random.Random(seed)
    rows = []
    per_pipe = {}
    for p in sorted(images_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        stem = p.stem
        pid = resolver.resolve(stem)
        if pid not in pipes:
            raise KeyError(f"pipes.csv 에 없는 배관 ID: {pid} (영상 {p.name})")
        lbl = labels_dir / (stem + ".txt")
        per_pipe.setdefault(pid, []).append((stem, str(p), str(lbl) if lbl.exists() else ""))
    for pid, items in per_pipe.items():
        rng.shuffle(items)
        if max_per_pipe:
            items = items[:max_per_pipe]
        if pid in test_pipes:
            for i, (stem, img, lbl) in enumerate(items):
                split = "adapt" if i < adapt_n else "test"
                rows.append([stem, img, lbl, pid, pipes[pid], split])
        elif pid == val_pipe:
            rows += [[stem, img, lbl, pid, pipes[pid], "val"] for stem, img, lbl in items]
        else:
            rows += [[stem, img, lbl, pid, pipes[pid], "train"] for stem, img, lbl in items]
    rows.sort(key=lambda r: (r[5], r[3], r[0]))
    return rows


def write_manifest(rows, out):
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "image", "label", "pipe_id", "diameter_mm", "split"])
        w.writerows(rows)


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["diameter_mm"] = float(r["diameter_mm"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pipes", required=True, help="pipes.csv (pipe_id, diameter_mm)")
    ap.add_argument("--resolver", default="regex", choices=["regex", "csv"])
    ap.add_argument("--pattern", default=r"^([A-Za-z]+)[_-]")
    ap.add_argument("--map-csv", default=None, help="resolver=csv 일 때 stem,pipe_id 매핑")
    ap.add_argument("--test-pipes", required=True, help="쉼표 구분, 예: K,E")
    ap.add_argument("--val-pipe", required=True)
    ap.add_argument("--adapt-n", type=int, default=200)
    ap.add_argument("--max-per-pipe", type=int, default=None, help="배관당 최대 영상 수(샘플링, 선택)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="manifest.csv")
    a = ap.parse_args()
    pipes = load_pipes(a.pipes)
    res = PipeResolver(a.resolver, a.pattern, a.map_csv)
    rows = build_manifest(a.images, a.labels, pipes, res, set(a.test_pipes.split(",")), a.val_pipe, a.adapt_n, a.seed, a.max_per_pipe)
    write_manifest(rows, a.out)
    from collections import Counter
    print(f"{a.out}: {len(rows)} 영상")
    print(Counter((r[5], r[3]) for r in rows))


if __name__ == "__main__":
    main()
