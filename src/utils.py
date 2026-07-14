import base64
import io
import json
import os
import random
import unicodedata
from typing import Dict, List, Literal

import faiss
import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )


def build_index(d: int = 768, index_type: Literal["hnsw", "flat_ip"] = "hnsw"):
    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    elif index_type == "flat_ip":
        index = faiss.IndexFlatIP(d)
    return index


def add_record_to_index(index: faiss.IndexHNSWFlat, embeddings: np.ndarray):
    index.add(embeddings)


def save_index(index: faiss.IndexHNSWFlat, metadata: List[Dict], target_dir: str):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(target_dir, "index.index"))
    with open(os.path.join(target_dir, "metadata.json"), "w") as f:
        json.dump({"metadata": metadata}, f)


def read_index(target_dir: str):
    index = faiss.read_index(os.path.join(target_dir, "index.index"))
    with open(os.path.join(target_dir, "metadata.json"), "r") as f:
        metadata = json.loads(f.read())

    return index, metadata


def encode_image(img_path: str, img_size: int = 336) -> str:
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((img_size, img_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()
