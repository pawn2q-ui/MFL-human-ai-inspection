# mfl_thesis — 관경 기반 공간 보정 + AI 판단보류 실험 코드 (v0.1, 2026-10-04)

합성 데이터로 끝까지 검증된 상태입니다. 실제 데이터에서는 **① 파일명/메타데이터 규칙(배관 ID)**과 **② pipes.csv(관경)** 두 가지만 맞추면 그대로 돌아갑니다.

## 0. 설치 (PC, 한 번만)
```
pip install numpy pandas opencv-python matplotlib scipy scikit-learn pyyaml
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA 버전은 nvidia-smi 확인 후 선택
pip install ultralytics
```
패키지 폴더(`mfl_thesis/`)가 들어 있는 상위 폴더에서 모든 명령을 `python -m mfl_thesis.xxx` 형태로 실행합니다.

## 1. 자체 검증 (데이터 없이, 5분)
```
python -m mfl_thesis.selftest --work selftest_out
```
"자체 검증 통과"가 나오면 환경이 정상입니다.

## 2. 실제 데이터 준비 (2주차)
1. `pipes.csv` 작성: 열 `pipe_id,diameter_mm` (논문 Table 5의 11개 배관 관경). 틀: `python -c "from mfl_thesis.metadata import write_pipes_template as w; w('pipes.csv', list('ABCDEFGHIJK'))"`
2. 영상 → 배관 ID 규칙 확인. 파일명에 배관 ID가 있으면 `--resolver regex --pattern "^([A-K])[_-]"`, 없으면 메타데이터로 `stem,pipe_id` CSV를 만들어 `--resolver csv --map-csv map.csv`
3. 매니페스트 생성 (시험 배관 2개 = 관경 범위 안 1개 + 범위 끝(최대 관경) 1개, 검증 1개, 적응용 200장/시험 배관):
```
python -m mfl_thesis.manifest --images <images_dir> --labels <labels_dir> --pipes pipes.csv \
    --resolver regex --pattern "^([A-K])[_-]" --test-pipes <P1>,<P2> --val-pipe <PV> --adapt-n 200 --out manifest.csv
```
4. 데이터 분석 (표 3-1, 그림 3-1, 3-2, 기타 통계):
```
python -m mfl_thesis.analyze_dataset --manifest manifest.csv --out analysis
```

## 3. 타일 캐시 (조건 A / B / C)
```
# 학습용 (양성 타일 전부 + 빈 타일 9:1)
python -m mfl_thesis.make_tiles --manifest manifest.csv --cond A --split train --mode train --out tiles/A_train
python -m mfl_thesis.make_tiles --manifest manifest.csv --cond A --split val   --mode train --out tiles/A_val
python -m mfl_thesis.make_tiles --manifest manifest.csv --cond B --split train --mode train --T 1280 --out tiles/B_train
python -m mfl_thesis.make_tiles --manifest manifest.csv --cond B --split val   --mode train --T 1280 --out tiles/B_val
python -m mfl_thesis.make_tiles --manifest manifest.csv --cond C --split train --mode train --T 1280 --out tiles/C_train
python -m mfl_thesis.make_tiles --manifest manifest.csv --cond C --split val   --mode train --T 1280 --out tiles/C_val
# 평가용 (모든 타일)
python -m mfl_thesis.make_tiles --manifest manifest.csv --cond A --split test,val --mode infer --out tiles/A_eval
python -m mfl_thesis.make_tiles --manifest manifest.csv --cond B --split test,val --mode infer --T 1280 --out tiles/B_eval
python -m mfl_thesis.make_tiles --manifest manifest.csv --cond C --split test,val --mode infer --T 1280 --out tiles/C_eval
python -m mfl_thesis.make_tiles --manifest manifest.csv --cond C --split test,val --mode infer --T 1280 --wrap 0 --out tiles/C_eval_nowrap   # 순환 패딩 비교
```
- `--s-star` 생략 시 학습 배관의 최소 s_y(= 어떤 영상도 축소하지 않음). `--T`는 검증 배관에서 1280/1920 중 선택.
- 먼저 `--limit 50`으로 소량 생성해 `tiles/C_train/images`의 타일과 라벨을 눈으로 확인할 것(17일 생명선).

## 4. 학습 (큐로 밤새, 19일 생명선)
`queue.txt`:
```
runs/A_s0/DONE | python -m mfl_thesis.train_condition --train tiles/A_train --val tiles/A_val --name A_s0 --seed 0 --epochs 60
runs/A_s1/DONE | python -m mfl_thesis.train_condition --train tiles/A_train --val tiles/A_val --name A_s1 --seed 1 --epochs 60
runs/B_s0/DONE | python -m mfl_thesis.train_condition --train tiles/B_train --val tiles/B_val --name B_s0 --seed 0 --epochs 40
runs/C_s0/DONE | python -m mfl_thesis.train_condition --train tiles/C_train --val tiles/C_val --name C_s0 --seed 0 --epochs 40
runs/B_s1/DONE | python -m mfl_thesis.train_condition --train tiles/B_train --val tiles/B_val --name B_s1 --seed 1 --epochs 40
runs/C_s1/DONE | python -m mfl_thesis.train_condition --train tiles/C_train --val tiles/C_val --name C_s1 --seed 1 --epochs 40
preds/A_s0/tile_stats.json | python -m mfl_thesis.infer_tiles --tiles tiles/A_eval --model runs/A_s0/weights/best.pt --out preds/A_s0
preds/A_s1/tile_stats.json | python -m mfl_thesis.infer_tiles --tiles tiles/A_eval --model runs/A_s1/weights/best.pt --out preds/A_s1
preds/B_s0/tile_stats.json | python -m mfl_thesis.infer_tiles --tiles tiles/B_eval --model runs/B_s0/weights/best.pt --out preds/B_s0
preds/B_s1/tile_stats.json | python -m mfl_thesis.infer_tiles --tiles tiles/B_eval --model runs/B_s1/weights/best.pt --out preds/B_s1
preds/C_s0/tile_stats.json | python -m mfl_thesis.infer_tiles --tiles tiles/C_eval --model runs/C_s0/weights/best.pt --out preds/C_s0
preds/C_s1/tile_stats.json | python -m mfl_thesis.infer_tiles --tiles tiles/C_eval --model runs/C_s1/weights/best.pt --out preds/C_s1
preds/C_s0_nowrap/tile_stats.json | python -m mfl_thesis.infer_tiles --tiles tiles/C_eval_nowrap --model runs/C_s0/weights/best.pt --out preds/C_s0_nowrap
```
```
python -m mfl_thesis.run_queue queue.txt
```
GPU 메모리 부족이면 `--batch 8`. 중단되면 같은 명령으로 재실행(완료된 항목은 건너뜀).

## 5. 평가 (4주차)
```
python -m mfl_thesis.evaluate --manifest manifest.csv --preds preds/C_s0/preds --split test --out eval/C_s0      # 조건·시드별 반복
python -m mfl_thesis.evaluate --manifest manifest.csv --preds preds/C_s0/preds --split val  --out eval/C_s0_val  # 검증 배관 보조 결과
```
산출: `metrics.json`(클래스별 recall·miss·precision·FP·AP50, 이음부), `pod_circ.csv`(배관별 a90, CI), `pod_bins_circ.csv`(mm 구간 POD), `pod_circ.png`, `gt_records.csv`.
두 조건 차이의 신뢰구간: `python -c "from mfl_thesis.evaluate import *; import pandas as pd; print(paired_bootstrap(pd.read_csv('eval/A_s0/gt_records.csv'), pd.read_csv('eval/C_s0/gt_records.csv')))"`

## 6. 판단보류 (4주차)
```
python -m mfl_thesis.deferral --manifest manifest.csv --preds preds/C_s0/preds --preds2 preds/C_s1/preds --split test --score sub_thr_max --lambda 1 --out defer/E_sub
python -m mfl_thesis.deferral --manifest manifest.csv --preds preds/C_s0/preds --preds2 preds/C_s1/preds --split test --score ensemble --out defer/E_ens
python -m mfl_thesis.deferral --manifest manifest.csv --preds preds/B_s0/preds --preds2 preds/B_s1/preds --split test --score sub_thr_max --lambda 1 --out defer/D_sub
```
산출: `deferral.json`(k=5/10/20% 잔여 미탐·포착률·박스 수·검사원 효과 r 민감도·배관별 배분, 무작위 기준, AUROC), `risk_coverage.png`.
점수 선택은 **검증 배관(`--split val`)** 결과로 하고 시험 배관 결과는 보고만 한다.

## 7. 소수 표본 적응 (4.5절)
```
python -m mfl_thesis.make_tiles --manifest manifest.csv --cond C --split adapt --mode train --T 1280 --out tiles/C_adapt
python -m mfl_thesis.train_condition --train tiles/C_adapt --val tiles/C_val --name C_s0_adapt200 --model runs/C_s0/weights/best.pt --epochs 5 --lr0 0.001 --freeze 10
```
(50장 적응은 manifest 를 `--adapt-n 50`으로 다시 만들거나 C_adapt 에서 50장만 골라 사용)

## 8. 시연 프로그램
```
python -m mfl_thesis.demo --image <영상.jpg> --diameter 711 --model runs/C_s0/weights/best.pt --s-star 0.598 --T 1280 --s-range 0.598,1.064 --out demo_out
```

## 파일 안내
| 파일 | 역할 | 논문 절 |
|---|---|---|
| config.py | 상수, s_y(D) | 2.1, 4.1 |
| metadata.py / manifest.py | 배관 ID·관경, 분할 | 3.1, 3.3 |
| analyze_dataset.py | 표 3-1, 그림 3-1/3-2 | 3.2 |
| geometry.py | 배율·타일·순환 패딩·역변환·NMS | 4.1, 4.2 |
| make_tiles.py | 조건 A/B/C 타일 캐시 | 4.2 |
| train_condition.py / run_queue.py | 학습·큐 | 3.4 |
| infer_tiles.py | 추론 → 원본 좌표 | 4.2 |
| evaluate.py | 미탐률·AP·POD/a90·부트스트랩 | 3.4, 5장 |
| deferral.py | 검토 점수·보류 평가 | 4.3, 4.4 |
| demo.py | 시연 | 경진대회 |
| synth_data.py / selftest.py | 합성 데이터·자체 검증 | — |
