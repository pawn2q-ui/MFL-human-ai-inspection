"""
순차 실행 큐: 여러 학습·추론 명령을 밤새 차례로 돌린다. 완료 표시 파일이 있으면 건너뛰므로 중단 후 재실행해도 이어서 돈다.

큐 파일(queue.txt) 예 — 한 줄에 "완료표시경로 | 명령":
  runs/A_s0/DONE | python -m mfl_thesis.train_condition --train tiles/A_train --val tiles/A_val --name A_s0 --seed 0
  runs/A_s1/DONE | python -m mfl_thesis.train_condition --train tiles/A_train --val tiles/A_val --name A_s1 --seed 1
  preds/A_s0/tile_stats.json | python -m mfl_thesis.infer_tiles --tiles tiles/A_eval --model runs/A_s0/weights/best.pt --out preds/A_s0

사용: python -m mfl_thesis.run_queue queue.txt
"""
import subprocess
import sys
import time
from pathlib import Path


def main():
    qfile = Path(sys.argv[1] if len(sys.argv) > 1 else "queue.txt")
    log = open("queue.log", "a", encoding="utf-8")
    for line in qfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        marker, cmd = [x.strip() for x in line.split("|", 1)]
        if Path(marker).exists():
            print(f"건너뜀(완료): {cmd}"); continue
        t0 = time.time(); print(f"시작: {cmd}"); log.write(f"[{time.ctime()}] START {cmd}\n"); log.flush()
        r = subprocess.run(cmd, shell=True)
        dt = (time.time() - t0) / 3600
        status = "OK" if r.returncode == 0 and Path(marker).exists() else f"FAIL(rc={r.returncode})"
        print(f"{status} ({dt:.2f} h): {cmd}"); log.write(f"[{time.ctime()}] {status} {dt:.2f}h {cmd}\n"); log.flush()
        if status != "OK":
            print("실패한 항목이 있어 다음으로 넘어갑니다. queue.log 확인.")


if __name__ == "__main__":
    main()
