"""
조건별 학습 래퍼 (ultralytics). 공식 프로토콜의 학습 조건을 그대로 따른다:
  YOLOv8s(본 연구) / COCO 사전학습 / imgsz 640 / rect / AMP / patience 20 / 양성 9:1 (타일 캐시에서 이미 적용)

사용 예
  python -m mfl_thesis.train_condition --train tiles/C_train --val tiles/C_val --name C_s0 --seed 0 --epochs 60 --batch 16
val 타일 캐시는 make_tiles --split val --mode train 으로 만든 것을 쓴다(검증 배관).

v0.2 (10-06): Windows에서 검증 로더 전환 시 'CUDA error: resource already mapped' / 가짜 OOM이 나는 문제 대응
  - 데이터로더 고정 메모리(pin_memory) 끔 (PIN_MEMORY=False), 워커 기본 2
  - 중단된 학습은 last.pt 에서 이어서 학습 (--no-resume 로 끔)
  - ultralytics 가 결과를 runs/detect/runs/<name> 아래에 두더라도 best.pt/last.pt/results.csv 를 runs/<name>/ 로 복사
"""
import argparse
import os
import shutil
from pathlib import Path
import yaml

# ultralytics 를 import 하기 전에 설정해야 효과가 있다
os.environ.setdefault("PIN_MEMORY", "False")

from .config import CLASSES


def write_data_yaml(train_dir, val_dir, out_yaml):
    d = {"path": "", "train": str(Path(train_dir).resolve() / "images"), "val": str(Path(val_dir).resolve() / "images"),
         "names": {i: c for i, c in enumerate(CLASSES)}}
    Path(out_yaml).parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(d, open(out_yaml, "w"), allow_unicode=True)
    return out_yaml


def candidate_save_dirs(project, name):
    """ultralytics 버전에 따라 결과 폴더 위치가 다르다. 가능한 위치를 모두 본다."""
    p = Path(project)
    return [p / name, Path("runs") / "detect" / p / name, Path("runs") / "detect" / name]


def find_last_pt(project, name):
    for d in candidate_save_dirs(project, name):
        f = d / "weights" / "last.pt"
        if f.exists():
            return f
    return None


def collect_outputs(project, name, save_dir):
    """best.pt / last.pt / results.csv 를 runs/<name>/ 로 모은다 (큐·추론 경로 고정용)."""
    dst = Path(project) / name
    (dst / "weights").mkdir(parents=True, exist_ok=True)
    sd = Path(save_dir)
    for w in ("best.pt", "last.pt"):
        src = sd / "weights" / w
        if src.exists() and src.resolve() != (dst / "weights" / w).resolve():
            shutil.copy2(src, dst / "weights" / w)
    for f in ("results.csv", "args.yaml"):
        src = sd / f
        if src.exists() and src.resolve() != (dst / f).resolve():
            shutil.copy2(src, dst / f)
    (dst / "save_dir.txt").write_text(str(sd.resolve()), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--project", default="runs")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr0", type=float, default=None, help="미세조정(4.5절) 시 본 학습의 1/10 등으로 지정")
    ap.add_argument("--freeze", type=int, default=None, help="미세조정 시 고정할 층 수 (예: 10 = 백본)")
    ap.add_argument("--no-resume", action="store_true", help="last.pt 가 있어도 처음부터 다시 학습")
    a = ap.parse_args()
    from ultralytics import YOLO

    data_yaml = write_data_yaml(a.train, a.val, Path(a.project) / a.name / "data.yaml")
    last = None if a.no_resume else find_last_pt(a.project, a.name)
    done = False
    if last is not None:
        try:
            print(f"이어서 학습: {last}")
            model = YOLO(str(last))
            model.train(resume=True)
            done = True
        except Exception as e:   # 체크포인트가 깨졌거나 이어받기가 안 되면 처음부터
            print(f"이어받기 실패({e}) → 처음부터 다시 학습")
    if not done:
        model = YOLO(a.model)
        kw = dict(data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers, device=a.device,
                  amp=True, pretrained=True, rect=True, patience=a.patience, seed=a.seed, deterministic=True,
                  project=a.project, name=a.name, exist_ok=True, verbose=True)
        if a.lr0 is not None:          # 미세조정: optimizer=auto 는 lr0 를 무시하므로 명시
            kw["lr0"] = a.lr0
            kw["optimizer"] = "AdamW"
            kw["warmup_epochs"] = 0
            kw["patience"] = max(a.patience, a.epochs)
        if a.freeze is not None:
            kw["freeze"] = a.freeze
        model.train(**kw)

    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None) or Path(a.project) / a.name
    collect_outputs(a.project, a.name, save_dir)
    Path(a.project, a.name, "DONE").write_text("ok")
    print(f"완료: {a.name} → {Path(a.project) / a.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
