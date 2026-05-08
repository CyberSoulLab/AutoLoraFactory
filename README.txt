# AutoLoRAFactory

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


# Project Structure

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

# Requirements

Environment
Python 3.10+
CUDA GPU
FFmpeg
Stable Diffusion WebUI
kohya_ss / sd-scripts
