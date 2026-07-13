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


def spherical_centroid(lats, lons):
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    x_mean = x.mean()
    y_mean = y.mean()
    z_mean = z.mean()

    lon_centroid = np.arctan2(y_mean, x_mean)
    hyp = np.sqrt(x_mean**2 + y_mean**2)
    lat_centroid = np.arctan2(z_mean, hyp)

    return np.degrees(lat_centroid), np.degrees(lon_centroid)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def normalize_text(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower()
