# AutoLoRA

> Automated Stable Diffusion LoRA pipeline with dataset extraction, training, generation, and evaluation.

AutoLoRA is a Python-based automation pipeline for creating high-quality LoRA models from videos.

It automatically:

- extracts frames from videos
- detects and crops faces
- removes duplicate images
- selects high-quality samples
- generates captions
- trains LoRA models
- generates images
- evaluates outputs with CLIP
- selects the best checkpoint automatically

---

# Features

## Dataset Automation

- Video frame extraction with FFmpeg
- Face detection using InsightFace
- Automatic face crop & resize
- Duplicate image removal
- Quality-based image filtering
- Automatic caption generation

## LoRA Training

- Fully automated LoRA training
- Hyperparameter search
- Automatic checkpoint evaluation
- Best model selection
- Batch experiment support

## Image Evaluation

- CLIP similarity scoring
- Simple aesthetic scoring
- Automatic ranking system

---

# Pipeline Overview

```text
Video (.mp4)
    ↓
Frame Extraction
    ↓
Face Detection & Crop
    ↓
Duplicate Removal
    ↓
Quality Filtering
    ↓
Caption Generation
    ↓
LoRA Training
    ↓
Image Generation
    ↓
CLIP Evaluation
    ↓
Best LoRA Selection
```

---

# Project Structure

```text
AutoLora/
├── input/                 # Input videos
├── dataset/
│   └── 10_person/         # Training dataset
├── lora/                  # Trained LoRA models
├── outputs/               # Generated images
├── best_lora/             # Best checkpoints
├── pipeline/
│   ├── extract.py
│   ├── train.py
│   ├── generate.py
│   ├── evaluate.py
│   ├── utils.py
│   └── run.py
└── results.json
```

---

# Requirements

## Environment

- Python 3.10+
- CUDA GPU
- FFmpeg
- Stable Diffusion WebUI
- kohya_ss / sd-scripts

## Python Packages

```bash
pip install \
  torch \
  torchvision \
  diffusers \
  transformers \
  insightface \
  opencv-python \
  pillow \
  tqdm \
  imagehash \
  xformers
```

---

# Setup

## 1. Install FFmpeg

Make sure FFmpeg is installed and added to PATH.

```bash
ffmpeg -version
```

---

## 2. Prepare Stable Diffusion Model

Place your base model here:

```text
C:/stable-diffusion-webui/models/Stable-diffusion/
```

Example:

```text
v1-5-pruned-emaonly.safetensors
```

---

## 3. Install sd-scripts

GitHub:

https://github.com/kohya-ss/sd-scripts

---

# Usage

## Step 1 — Add Videos

Place `.mp4` files inside:

```text
input/
```

---

## Step 2 — Generate Dataset

```bash
python pipeline/extract.py
```

This step:

- extracts frames
- detects faces
- crops faces
- removes duplicates
- filters low-quality images
- generates captions

---

## Step 3 — Train LoRA

```bash
python pipeline/train.py \
  --dim 16 \
  --lr 5e-5 \
  --steps 1000 \
  --name test_lora
```

---

## Step 4 — Generate Images

```bash
python pipeline/generate.py --name test_lora
```

Outputs:

```text
outputs/test_lora/
```

---

## Step 5 — Automatic Search

```bash
python pipeline/run.py
```

This automatically runs:

1. training
2. image generation
3. evaluation
4. best model selection

---

# Hyperparameter Search

Edit `SEARCH` in `run.py`.

```python
SEARCH = {
    "dim": [8, 16, 32],
    "lr": [1e-4, 5e-5],
    "steps": [1000, 2000]
}
```

---

# Evaluation Formula

```python
score = CLIP * 0.7 + aesthetic * 0.3
```

Current metrics:

- CLIP similarity score
- simple brightness-based aesthetic score

---

# Example Workflow

```bash
python pipeline/extract.py
python pipeline/run.py
```

That's it.

The pipeline handles the rest automatically.

---

# Notes

- CUDA environment required
- Windows paths are currently hardcoded
- Designed for personal research / experimentation
- Path cleanup and config externalization recommended

---

# Future Improvements

- BLIP automatic captioning
- better aesthetic predictor
- multi-character support
- automatic tagging
- DreamBooth support
- Web UI
- distributed training

---

# License

MIT License

---

# Author

CyberSoul Wing

---

# Disclaimer

This project was built in 2 days.

I do not accept ugly code.  
I will probably not maintain this repository.

Use it, modify it, break it — do whatever you want.
