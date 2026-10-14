"""
PipeMFL-240K test_data 다운로드 + 분할 tar 스트리밍 추출
(Windows / Linux 공통, 조각을 하나로 합치지 않으므로 추가 디스크 110GB가 필요 없음)

사용 전 준비:
    pip install -U huggingface_hub
    (대용량 다운로드 가속)  pip install hf_transfer

실행:
    python 01_download_and_extract.py --out D:/pipemfl        # 다운로드 + 추출
    python 01_download_and_extract.py --out D:/pipemfl --skip-download   # 이미 받았으면 추출만
"""
import argparse, os, tarfile, sys
from pathlib import Path

REPO = "PipeMFL/PipeMFL-240K"
SPLIT = "test_data"            # 4조각, 약 110GB
PARTS = [f"{SPLIT}/test_part_{i:02d}" for i in range(4)]


class ChainedReader:
    """여러 파일을 하나의 연속 스트림처럼 읽어 tar에 넘긴다."""
    def __init__(self, paths):
        self.paths = list(paths)
        self.idx = 0
        self.f = open(self.paths[0], "rb")
        self.total = sum(os.path.getsize(p) for p in self.paths)
        self.done = 0

    def read(self, n=-1):
        buf = b""
        while n < 0 or len(buf) < n:
            chunk = self.f.read(n - len(buf) if n >= 0 else 1 << 24)
            if chunk:
                buf += chunk
                self.done += len(chunk)
            else:
                self.f.close()
                self.idx += 1
                if self.idx >= len(self.paths):
                    break
                self.f = open(self.paths[self.idx], "rb")
        return buf

    def close(self):
        self.f.close()


def download(out: Path):
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=REPO, repo_type="dataset",
        allow_patterns=[f"{SPLIT}/*"],
        local_dir=str(out / "raw"),
        max_workers=4,
    )


def extract(out: Path):
    paths = [out / "raw" / p for p in PARTS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f"누락된 조각: {missing}")
    dest = out / SPLIT
    dest.mkdir(parents=True, exist_ok=True)
    rd = ChainedReader(paths)
    n = 0
    with tarfile.open(fileobj=rd, mode="r|*") as tf:
        for m in tf:
            tf.extract(m, path=dest)
            n += 1
            if n % 2000 == 0:
                print(f"  {n}개 추출, {rd.done / rd.total * 100:.1f}%  (최근: {m.name})")
    rd.close()
    print(f"완료: {n}개 파일 → {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="데이터를 둘 폴더 (예: D:/pipemfl)")
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.skip_download:
        print("다운로드 시작 (약 110GB, 중단돼도 다시 실행하면 이어받기 됨)")
        download(out)
    print("추출 시작")
    extract(out)
