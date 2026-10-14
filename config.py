"""
공통 상수와 환산식.
- PipeMFL 영상: 가로 5,000 px = 축 방향 5 m (1 mm/px), 세로 2,400 px = 배관 둘레 전체
- 세로 1 px의 실제 길이 s_y(D) = pi*D/2400 (D: 공칭 외경 mm)
"""
import math

IMG_W = 5000
IMG_H = 2400
AXIAL_MM_PER_PX = 1.0          # 5,000 mm / 5,000 px

# 공식 평가 코드의 클래스 순서 (metrics_yolo.py 기준)
CLASSES = ["MTL", "TEE", "BND", "CRC", "BRN", "GWA", "SWA", "ESP", "VAL", "FLA", "CAS", "SLE"]
DAMAGE = ["MTL", "CRC", "GWA", "SWA"]           # 손상 4종 (주요 평가 대상)
DAMAGE_IDS = [CLASSES.index(c) for c in DAMAGE]

# 공식 프로토콜 (조건 A)
A_WINDOW = 2400           # 2400x2400 창
A_TRAIN_X0 = [0, IMG_W - A_WINDOW]            # 학습: 창 2개 (좌/우)
A_INFER_STRIDE = 1300                          # 추론: SAHI 간격 1300 px (45.8% 겹침)
A_CONF = 0.25
NMS_IOU = 0.5
MATCH_IOU = 0.5           # 평가 시 정답-예측 대응 IoU

DETECTOR_INPUT = 640      # 탐지기 입력 크기


def s_y(diameter_mm: float) -> float:
    """세로 1픽셀이 나타내는 원주 방향 실제 길이 (mm/px)."""
    return math.pi * diameter_mm / IMG_H


def circumference_mm(diameter_mm: float) -> float:
    return math.pi * diameter_mm
