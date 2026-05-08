import subprocess
import sys
import itertools
import time
import shutil
import os
import json
import glob

from pipeline.evaluate import evaluate_folder

# =========================
# ■ 設定
# =========================
SEARCH = {
    "dim": [16],
    "lr": [5e-5],
    "steps": [1000]
}

PROMPT = "1girl, realistic, detailed face, cinematic lighting, 50mm lens"

RESULTS_FILE = "results.json"
BEST_DIR = "best_lora"

os.makedirs(BEST_DIR, exist_ok=True)

results = []

# =========================
# ■ step別評価
# =========================
def evaluate_steps(base_name):
    lora_files = glob.glob(f"lora/{base_name}-*.safetensors")

    best_score = -1
    best_name = None

    for path in lora_files:
        step_name = os.path.basename(path).replace(".safetensors", "")

        print(f"\n🔍 TEST: {step_name}")

        # ===== GENERATE =====
        subprocess.run([
            sys.executable, "pipeline/generate.py",
            "--name", step_name
        ], check=True)

        # ===== EVALUATE =====
        score = evaluate_folder(f"outputs/{step_name}", PROMPT)

        print(f"⭐ SCORE: {score}")

        if score > best_score:
            best_score = score
            best_name = step_name

    return best_name, best_score


# =========================
# ■ メインループ
# =========================
total = len(SEARCH["dim"]) * len(SEARCH["lr"]) * len(SEARCH["steps"])
count = 0

for dim, lr, steps in itertools.product(
    SEARCH["dim"], SEARCH["lr"], SEARCH["steps"]
):
    count += 1
    name = f"dim{dim}_lr{lr}"

    print("\n===================================")
    print(f"🚀 [{count}/{total}] START: {name}")
    print("===================================")

    t0 = time.time()

    # ===== TRAIN =====
    subprocess.run([
        sys.executable, "pipeline/train.py",
        "--dim", str(dim),
        "--lr", str(lr),
        "--steps", str(steps),
        "--name", name
    ], check=True)

    print(f"✅ TRAIN DONE ({round(time.time()-t0,1)}s)")

    # ===== STEP別評価 =====
    best_name, best_score = evaluate_steps(name)

    print(f"\n🏆 BEST STEP: {best_name} ({best_score})")

    # ===== bestコピー =====
    if best_name:
        src = f"lora/{best_name}.safetensors"
        dst = os.path.join(BEST_DIR, f"{best_name}.safetensors")

        if os.path.exists(src):
            shutil.copy(src, dst)

    # ===== 結果保存 =====
    results.append({
        "base_name": name,
        "best_model": best_name,
        "score": best_score,
        "dim": dim,
        "lr": lr,
        "steps": steps
    })

    results.sort(key=lambda x: x["score"], reverse=True)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"📊 CURRENT BEST: {results[0]['best_model']} ({results[0]['score']})")

print("\n🎯 ALL DONE")