import os
from typing import Literal

import numpy as np
import polars as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


SCENE2ID = {"indoor": 0, "outdoor": 1}
VIEWPOINT2ID = {"ground": 0, "aerial": 1}


class ImageDataset(Dataset):
    def __init__(self, df: pl.DataFrame, base_img_path: str, transform=None):
        super().__init__()

        self.df = df
        self.transform = transform
        self.base_img_path = base_img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        image = Image.open(
            os.path.join(self.base_img_path, row["dataset"], row["src"], row["id"] + ".jpg")
        ).convert("RGB")
        scene_label = SCENE2ID[row["scene_type"]]
        viewpoint_label = VIEWPOINT2ID[row["viewpoint_type"]]
        if self.transform:
            image = self.transform(image)
        return image, scene_label, viewpoint_label


class FlickrDataset(Dataset):
    """Like ImageDataset but for unlabeled data: does not read/require the
    scene_type or viewpoint_type columns, only what's needed to locate the
    image file."""

    def __init__(self, df: pl.DataFrame, base_img_path: str, transform=None):
        super().__init__()

        self.df = df
        self.transform = transform
        self.base_img_path = base_img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        image = Image.open(
            os.path.join(self.base_img_path, row["img_id"])
        ).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def compute_class_weights(df: pl.DataFrame):
    """Weights samples by their joint (scene_type, viewpoint_type) combo so
    that all four combinations are represented evenly during sampling."""
    joint_labels = list(zip(df["scene_type"].to_list(), df["viewpoint_type"].to_list()))
    joint2id = {v: i for i, v in enumerate(sorted(set(joint_labels)))}
    label_ids = np.array([joint2id[v] for v in joint_labels])
    class_counts = np.bincount(label_ids, minlength=len(joint2id))
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    sample_weights = class_weights[label_ids]
    return class_weights, sample_weights


def _build_transform(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def make_dataloader(
    df: pl.DataFrame,
    base_img_path: str,
    batch_size: int,
    split: Literal["train", "val", "test"],
    sampler=None,
):
    transform_fn = _build_transform(train=split == "train")

    dataset = ImageDataset(df, base_img_path, transform_fn)
    return DataLoader(
        dataset,
        batch_size,
        shuffle=split == "train" and sampler is None,
        sampler=sampler,
        num_workers=4
    )


def make_inference_dataloader(df: pl.DataFrame, base_img_path: str, batch_size: int):
    """Dataloader for real, unlabeled inference data — no scene_type or
    viewpoint_type columns required."""
    dataset = FlickrDataset(df, base_img_path, _build_transform(train=False))
    return DataLoader(dataset, batch_size, shuffle=False, num_workers=4)
