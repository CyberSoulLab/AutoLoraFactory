import subprocess
import sys
from tqdm import tqdm


def run_command(cmd, title):
    print(f"\n {title} START")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        bufsize=1
    )

    pbar = tqdm(desc=title, unit="line")

    for line in process.stdout:
        # ログ出しは軽く（速度維持）
        if line.strip():
            print(line.strip(), flush=False)
        pbar.update(1)

    process.wait()
    pbar.close()

    if process.returncode != 0:
        raise RuntimeError(f"{title} FAILED")

    print(f" {title} DONE")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default="16")
    parser.add_argument("--lr", default="1e-4")
    parser.add_argument("--steps", default="1000")
    parser.add_argument("--name", required=True)

    args = parser.parse_args()

    PYTHON = sys.executable

    # =========================
    # ■ TRAIN
    # =========================
    train_cmd = [
        PYTHON,
        "C:/sd-scripts/train_network.py",
        "--pretrained_model_name_or_path",
        "C:/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors",
        "--train_data_dir",
        "dataset",
        "--output_dir",
        "lora",
        "--output_name",
        args.name,
        "--resolution",
        "512,512",
        "--train_batch_size",
        "1",

        "--network_module=networks.lora",
        "--network_dim",
        args.dim,
        "--network_alpha",
        args.dim,

        "--learning_rate",
        args.lr,
        "--max_train_steps",
        args.steps,

        "--lr_scheduler",
        "constant",
        "--optimizer_type",
        "AdamW",

        "--save_every_n_steps",
        "200",

        "--caption_extension",
        ".txt",
        "--shuffle_caption",

        "--enable_bucket",
        "--min_bucket_reso",
        "256",
        "--max_bucket_reso",
        "1024",

        "--seed",
        "42",

        "--save_model_as",
        "safetensors",
        "--mixed_precision",
        "fp16",
        "--cache_latents",
        "--xformers"
    ]

    run_command(train_cmd, "TRAIN")

    # =========================
    # ■ GENERATE
    # =========================
    gen_cmd = [
        PYTHON,
        "pipeline/generate.py",
        "--name",
        args.name
    ]

    run_command(gen_cmd, "GENERATE")

    # =========================
    # ■ EVALUATE
    # =========================
    eval_cmd = [
        PYTHON,
        "pipeline/evaluate.py",
        "--name",
        args.name
    ]

    run_command(eval_cmd, "EVALUATE")