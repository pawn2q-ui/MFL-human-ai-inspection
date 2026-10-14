"""
배관 메타데이터와 영상→배관 매핑.

데이터셋의 실제 폴더 구조와 파일명 규칙은 다운로드 후 확인해야 하므로,
영상 파일명(stem)에서 배관 ID를 얻는 방법을 두 가지로 둔다.
  1) regex 모드: 파일명에 배관 ID가 들어 있을 때   (예: "A_000123.jpg" -> "A")
  2) csv   모드: 영상 stem -> pipe_id 매핑 CSV가 있을 때 (열: stem, pipe_id)

배관별 공칭 외경은 pipes.csv (열: pipe_id, diameter_mm) 로 관리한다.
"""
import csv
import re
from pathlib import Path
from typing import Dict, Optional


def load_pipes(csv_path) -> Dict[str, float]:
    """pipes.csv -> {pipe_id: diameter_mm}"""
    pipes = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["pipe_id"].strip()
            if not pid:
                continue
            pipes[pid] = float(row["diameter_mm"])
    if not pipes:
        raise ValueError(f"{csv_path}: 배관 정보가 비어 있습니다")
    return pipes


class PipeResolver:
    """영상 stem -> pipe_id"""

    def __init__(self, mode: str = "regex", pattern: str = r"^([A-Za-z]+)[_-]", csv_path: Optional[str] = None):
        self.mode = mode
        self.pattern = re.compile(pattern) if pattern else None
        self.map = {}
        if mode == "csv":
            if not csv_path:
                raise ValueError("csv 모드에는 csv_path가 필요합니다")
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.map[row["stem"].strip()] = row["pipe_id"].strip()

    def resolve(self, stem: str) -> str:
        if self.mode == "csv":
            if stem not in self.map:
                raise KeyError(f"매핑 CSV에 없는 영상: {stem}")
            return self.map[stem]
        m = self.pattern.match(stem)
        if not m:
            raise ValueError(f"파일명에서 배관 ID를 찾지 못함: {stem} (pattern={self.pattern.pattern})")
        return m.group(1)


def write_pipes_template(path, pipe_ids, diameters=None):
    """pipes.csv 틀 생성 (diameters가 없으면 빈칸으로 두어 사람이 채움)."""
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pipe_id", "diameter_mm"])
        for pid in pipe_ids:
            w.writerow([pid, "" if diameters is None else diameters.get(pid, "")])
    return path
