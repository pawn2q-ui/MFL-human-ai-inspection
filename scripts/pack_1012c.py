"""queue7 결과(루프 2·3회차, 시연 점검)만 묶는다. 패키지 폴더에서:  python pack_1012c.py  →  results_1013_loop.zip"""
import os, zipfile

def add(z, path):
    if os.path.isfile(path):
        z.write(path, path); return 1
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f); z.write(p, p); n += 1
    return n

items = ["eval_sel2", "eval_sel3", "eval_full2", "eval_full3", "analysis/sel_A2.txt", "analysis/sel_A3.txt",
         "manifest_selA2.csv", "manifest_selA3.csv", "demo_out", "queue.log"]
missing = []
with zipfile.ZipFile("results_1013_loop.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in items:
        if os.path.exists(p):
            add(z, p)
        else:
            missing.append(p)
print("results_1013_loop.zip: %.1f MB" % (os.path.getsize("results_1013_loop.zip") / 1e6))
print("없는 항목:", ", ".join(missing) if missing else "없음 (모두 포함)")
