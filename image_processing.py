import os, glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import timm
from timm.data import resolve_data_config, create_transform
from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"

# Jori, pasikeisk modeli i "resnet50" ir embed_property nusiimk paths = paths[:max_photos]
img_model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=0)
img_model.eval().to(device)

config = resolve_data_config({}, model=img_model)
transform = create_transform(**config)

def embed_one_image(path):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = img_model(x).cpu().numpy().squeeze()  # ~2048 dims
    return emb

def embed_property(reference, images_root="images", max_photos=8):
    folder = os.path.join(images_root, str(reference))
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    if len(paths) == 0:
        return None

    paths = paths[:max_photos] # limit for speed

    embs = [embed_one_image(p) for p in paths]
    embs = np.vstack(embs)
    return embs.mean(axis=0)   # 1 vector per property
