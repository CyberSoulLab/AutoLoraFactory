AutoLoRA

動画から自動で顔データセットを生成し、LoRA学習・画像生成・評価までを一括で実行する Python パイプライン。

Stable Diffusion + LoRA の実験を高速化するための個人研究用ツールです。

Features
動画からフレーム抽出
顔検出 & 自動クロップ
重複画像削除
高品質画像のみ選別
自動キャプション生成
LoRA学習自動化
LoRA画像生成
CLIPベース自動評価
ハイパーパラメータ探索
ベストLoRA自動保存
Pipeline
動画(.mp4)
   ↓
FFmpegフレーム抽出
   ↓
顔検出 & クロップ
   ↓
重複除去
   ↓
高品質画像選別
   ↓
キャプション生成
   ↓
LoRA学習
   ↓
画像生成
   ↓
CLIP評価
   ↓
Best LoRA 保存
Directory Structure
AutoLora/
├── input/                 # 入力動画
├── dataset/
│   └── 10_person/         # 学習用データセット
├── lora/                  # 学習済LoRA
├── outputs/               # 生成画像
├── best_lora/             # ベストLoRA
├── pipeline/
│   ├── extract.py
│   ├── train.py
│   ├── generate.py
│   ├── evaluate.py
│   ├── utils.py
│   └── run.py
└── results.json
Requirements
Python 3.10+
CUDA GPU
FFmpeg
Stable Diffusion WebUI
kohya_ss / sd-scripts
Python Libraries
pip install torch torchvision diffusers transformers insightface opencv-python pillow tqdm imagehash xformers
Setup
1. FFmpeg インストール

FFmpeg を PATH に追加してください。

確認:

ffmpeg -version
2. Stable Diffusion Model

以下を配置:

C:/stable-diffusion-webui/models/Stable-diffusion/

使用モデル:

v1-5-pruned-emaonly.safetensors
3. sd-scripts

以下を導入:

sd-scripts (kohya_ss)

Usage
Step 1 — 動画を配置
input/
 └── sample.mp4
Step 2 — データセット生成
python pipeline/extract.py

実行内容:

フレーム抽出
顔検出
自動クロップ
重複削除
品質選別
キャプション生成
Step 3 — 学習
python pipeline/train.py \
  --dim 16 \
  --lr 5e-5 \
  --steps 1000 \
  --name test_lora
Step 4 — 生成
python pipeline/generate.py --name test_lora

生成画像:

outputs/test_lora/
Step 5 — 自動探索
python pipeline/run.py

自動実行:

学習
生成
評価
ベストモデル選定
Hyperparameter Search

run.py

SEARCH = {
    "dim": [16],
    "lr": [5e-5],
    "steps": [1000]
}

例:

SEARCH = {
    "dim": [8,16,32],
    "lr": [1e-4,5e-5],
    "steps": [1000,2000]
}
Evaluation

評価スコア:

score = CLIP * 0.7 + aesthetic * 0.3

現在:

CLIP similarity
簡易 aesthetic score

を使用。

Notes
CUDA 環境前提
Windows パス固定あり
個人研究用途
実運用前にパス整理推奨
Future Improvements
BLIP captioning
本格 aesthetic predictor
multi-character support
automatic tagging
DreamBooth support
distributed training
web UI
License

MIT

Author

CyberSoul Wing