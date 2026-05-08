import os
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline
import argparse

# =========================
# ■ 設定
# =========================
BASE_MODEL = r"C:/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors"
LORA_DIR = "lora"
OUTPUT_DIR = "outputs"

DEVICE = "cuda"
DTYPE = torch.float16

PROMPT = "(masterpiece, best quality:1.2), <lora:my_lora:0.6>, 1girl, realistic, skin texture, detailed face, cinematic lighting, 50mm lens, depth of field, photorealistic"
NEGATIVE_PROMPT = "(worst quality, low quality:1.4), bad anatomy, bad hands, extra fingers, missing fingers, blurry, jpeg artifacts, text, watermark"

NUM_IMAGES = 5
STEPS = 20
GUIDANCE = 7.5


# =========================
# ■ パイプラインロード（1回だけ）
# =========================
def load_pipe(lora_path=None):
    print(" PIPELINE LOAD")

    pipe = StableDiffusionPipeline.from_single_file(
        BASE_MODEL,
        torch_dtype=DTYPE,
        safety_checker=None
    ).to(DEVICE)

    # メモリ最適化
    pipe.enable_xformers_memory_efficient_attention()

    if lora_path:
        print(f" LoRA LOAD: {lora_path}")
        pipe.load_lora_weights(lora_path)

    return pipe


# =========================
# ■ 生成処理（軽量ループ）
# =========================
def generate(pipe, name):
    print(f"\n GENERATE START: {name}")

    out_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    for i in tqdm(range(NUM_IMAGES), desc="GENERATE", unit="img"):
        image = pipe(
            PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE
        ).images[0]

        save_path = os.path.join(out_dir, f"{i:05d}.png")
        image.save(save_path)

    print(" GENERATE DONE")


# =========================
# ■ 複数LoRA対応（必要なら拡張）
# =========================
def run(name):
    lora_path = os.path.join(LORA_DIR, f"{name}.safetensors")

    print(f" LoRA PATH: {lora_path}")

    if not os.path.exists(lora_path):
        print(" LoRAが存在しない")
        return

    # ★ここが重要：1回だけロード
    pipe = load_pipe(lora_path)

    # 生成ループ
    generate(pipe, name)


# =========================
# ■ エントリーポイント
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    run(args.name)