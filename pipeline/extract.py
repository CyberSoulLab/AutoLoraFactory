import os
import subprocess
import cv2
import imagehash
from PIL import Image
from insightface.app import FaceAnalysis
from tqdm import tqdm
import numpy as np
import logging

# =========================
# ■ 設定（絶対変えるな）
# =========================
BASE_DIR = r"AutoLoraPath"
INPUT_DIR = os.path.join(BASE_DIR, "input")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "10_person")

FPS = 0.2
TOP_K = 100
PAD = 0.3

cv2.setNumThreads(1)

# =========================
# ■ ログ抑制（邪魔ログ消す）
# =========================
logging.getLogger("insightface").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# =========================
# ■ GPU固定（ここ重要）
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider"]  # ★ CPU fallback禁止
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

# =========================
# ■ リサイズ（歪み防止）
# =========================
def resize_with_padding(img, size=512):
    h, w = img.shape[:2]

    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas

# =========================
# ■ FFmpeg（dataset直書き）
# =========================
def extract_frames(video_path):
    os.makedirs(DATASET_DIR, exist_ok=True)

    output_pattern = os.path.join(DATASET_DIR, "%05d.png")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-hwaccel", "cuda",
        "-c:v", "h264_cuvid",
        "-i", video_path,
        "-vf", f"fps={FPS}",
        "-progress", "pipe:1",
        "-nostats",
        output_pattern
    ]

    print(f"\n▶ 抽出: {os.path.basename(video_path)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    pbar = tqdm(desc="FFmpeg", unit="frame")

    for line in process.stdout:
        if "frame=" in line:
            pbar.update(1)

    process.wait()
    pbar.close()

# =========================
# ■ 顔抽出
# =========================
def filter_and_crop_faces(dir_path):
    files = [f for f in os.listdir(dir_path) if f.endswith(".png")]

    keep, delete = 0, 0

    print(f"\n 顔抽出 START ({len(files)}枚)")

    for f in tqdm(files, desc="顔抽出"):
        path = os.path.join(dir_path, f)

        img = cv2.imread(path)
        if img is None:
            os.remove(path)
            continue

        faces = app.get(img)

        if len(faces) == 0:
            os.remove(path)
            delete += 1
            continue

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = map(int, face.bbox)

        h, w = img.shape[:2]
        fw = x2 - x1
        fh = y2 - y1

        x1 = int(max(0, x1 - fw * PAD))
        y1 = int(max(0, y1 - fh * PAD))
        x2 = int(min(w, x2 + fw * PAD))
        y2 = int(min(h, y2 + fh * PAD))

        face_img = img[y1:y2, x1:x2]
        face_img = resize_with_padding(face_img)

        cv2.imwrite(path, face_img)
        keep += 1

    print(f" 顔抽出 DONE / 残り:{keep} 削除:{delete}")

# =========================
# ■ 重複削除
# =========================
def remove_duplicates(dir_path, threshold=8):
    files = [f for f in os.listdir(dir_path) if f.endswith(".png")]

    hashes = []
    removed = 0

    print(f"\n 重複削除 START ({len(files)}枚)")

    for f in tqdm(files, desc="重複削除"):
        path = os.path.join(dir_path, f)

        try:
            img = Image.open(path)
            h = imagehash.phash(img)
        except:
            continue

        if any(abs(h - existing) <= threshold for existing in hashes):
            os.remove(path)
            removed += 1
        else:
            hashes.append(h)

    print(f" 重複削除 DONE / 削除:{removed} 残り:{len(hashes)}")

# =========================
# ■ スコア選別
# =========================
def calc_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def select_best_faces(dir_path):
    files = [f for f in os.listdir(dir_path) if f.endswith(".png")]

    scored = []

    print(f"\n スコア選別 START ({len(files)}枚)")

    for f in tqdm(files, desc="スコア計算"):
        path = os.path.join(dir_path, f)
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        score = (h*w) + calc_sharpness(img)*10
        scored.append((score, path))

    scored.sort(reverse=True)

    for _, path in tqdm(scored[TOP_K:], desc="削除"):
        os.remove(path)

    print(f" スコア選別 DONE / 残り:{TOP_K} 削除:{max(0, len(scored)-TOP_K)}")

# =========================
# ■ リネーム
# =========================
def rename_files(dir_path):
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".png")])

    print(f"\n リネーム START ({len(files)}枚)")

    for i, f in enumerate(files):
        new_name = f"{i:05d}.png"
        os.rename(os.path.join(dir_path, f), os.path.join(dir_path, new_name))

    print(" リネーム DONE")

# =========================
# ■ キャプション
# =========================
def create_caption(dir_path, trigger="akari woman"):
    files = [f for f in os.listdir(dir_path) if f.endswith(".png")]

    print(f"\n✏️ キャプション START ({len(files)}枚)")

    for f in tqdm(files, desc="キャプション"):
        txt_path = os.path.join(dir_path, f.replace(".png", ".txt"))
        with open(txt_path, "w", encoding="utf-8") as fp:
            fp.write(trigger)

    print(" キャプション DONE")

# =========================
# ■ メイン
# =========================
def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    # ★ スキップ条件
    existing = [f for f in os.listdir(DATASET_DIR) if f.endswith(".png")]
    if len(existing) >= TOP_K:
        print(" dataset既に存在するためスキップ")
        return

    videos = [f for f in os.listdir(INPUT_DIR) if f.endswith(".mp4")]

    print(f"\n 動画数: {len(videos)}")

    # ① 全動画から抽出
    for v in videos:
        full = os.path.join(INPUT_DIR, v)
        extract_frames(full)

    # ② まとめて処理（元仕様）
    filter_and_crop_faces(DATASET_DIR)
    remove_duplicates(DATASET_DIR)
    select_best_faces(DATASET_DIR)

    rename_files(DATASET_DIR)
    create_caption(DATASET_DIR)

    print("\n🎉 dataset生成 完了")

if __name__ == "__main__":
    main()
