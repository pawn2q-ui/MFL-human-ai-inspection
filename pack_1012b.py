"""queue6 결과만 골라 작은 zip 2개로 묶는다. 패키지 폴더(mfl_thesis_PC_bundle)에 넣고:  python pack_1012b.py
  results_1012b_1.zip : 숫자 결과 (판단보류 A·점수 기준선, A 루프 평가, 오탐 분석)  — 보통 10 MB 안팎
  results_1012b_2.zip : 실패 사례 그림 (analysis/cases_*)                         — 10~20 MB
둘 다 채팅에 올리면 되고, 2번이 너무 크면 1번만 먼저 올려도 된다.
"""
import glob, os, zipfile

def add(z, path, skip=lambda p: False):
    if os.path.isfile(path):
        if not skip(path):
            z.write(path, path)
        return 1
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not skip(p):
                z.write(p, p); n += 1
    return n

part1 = ["defer/DA_sub", "defer/DA_npred", "defer/DA_combo", "defer/DA_ens", "defer/DA_rand", "defer/DA_sub_val", "defer/DA_combo_val",
         "defer/D_npred", "defer/D_combo", "defer/E_npred", "defer/E_combo", "defer/Ec_npred", "defer/Ec_combo", "defer/Ep_npred", "defer/Ep_combo", "defer/E_combo_val",
         "eval_sel/A_s1", "eval_sel/A_s1_ad200", "eval_sel/A_s1_adsel200",
         "eval_full/A_s1_ad200", "eval_full/A_s1_adsel200", "eval_full/A_s1_sel",
         "analysis/fp_analysis.csv", "analysis/sel_A.txt", "manifest_selA.csv", "queue.log"]
missing = []
with zipfile.ZipFile("results_1012b_1.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in part1:
        if not os.path.exists(p):
            missing.append(p); continue
        add(z, p, skip=lambda q: q.endswith("image_scores_lambda1.csv"))   # λ=1 영상별 표는 deferral.json 요약으로 충분
with zipfile.ZipFile("results_1012b_2.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    n = 0
    for d in glob.glob("analysis/cases_*"):
        n += add(z, d)
print("results_1012b_1.zip: %.1f MB" % (os.path.getsize("results_1012b_1.zip") / 1e6))
print("results_1012b_2.zip: %.1f MB (사례 그림 %d개)" % (os.path.getsize("results_1012b_2.zip") / 1e6, n))
if missing:
    print("없는 항목(아직 안 돌았거나 실패):", ", ".join(missing))
else:
    print("queue6 산출물이 모두 있습니다.")
