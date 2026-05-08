import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def clip_score(image, prompt):
    inputs = clip_processor(
        text=[prompt],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        return outputs.logits_per_image[0][0].item()


def aesthetic_score(image):
    # 簡易版（平均輝度ベース）
    # → 後で本物に差し替え可能
    return image.convert("L").resize((64, 64)).getextrema()[1]


def evaluate_folder(folder, prompt):
    scores = []

    for file in os.listdir(folder):
        if not file.endswith(".png"):
            continue

        path = os.path.join(folder, file)
        image = Image.open(path).convert("RGB")

        c = clip_score(image, prompt)
        a = aesthetic_score(image)

        score = c * 0.7 + a * 0.3

        scores.append(score)

    if not scores:
        return 0

    return sum(scores) / len(scores)