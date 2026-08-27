import os
from typing import Literal

import numpy as np
import polars as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


SCENE2ID = {"interior": 0, "exterior": 1}
VIEWPOINT2ID = {"ground": 0, "aerial": 1}

ID2SCENE = {v: k for k, v in SCENE2ID.items()}
ID2VIEWPOINT = {v: k for k, v in VIEWPOINT2ID.items()}


class ImageDataset(Dataset):
    def __init__(
        self, df: pl.DataFrame, base_img_path: str, transform=None, use_viewpoint: bool = False
    ):
        super().__init__()

        self.df = df
        self.transform = transform
        self.base_img_path = base_img_path
        self.use_viewpoint = use_viewpoint

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        image = Image.open(
            os.path.join(self.base_img_path, row["src"], row["id"] + ".jpg")
        ).convert("RGB")
        scene_label = SCENE2ID[row["label"]]
        if self.transform:
            image = self.transform(image)
        if self.use_viewpoint:
            viewpoint_label = VIEWPOINT2ID[row["viewpoint_type"]]
            return image, scene_label, viewpoint_label
        return image, scene_label


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


def compute_class_weights(df: pl.DataFrame, use_viewpoint: bool = False):
    """Weights samples by their label so that classes are represented evenly
    during sampling. When `use_viewpoint` is True, weights by the joint
    (label, viewpoint_type) combo instead so all four combinations are
    represented evenly."""
    if use_viewpoint:
        joint_labels = list(zip(df["label"].to_list(), df["viewpoint_type"].to_list()))
    else:
        joint_labels = df["label"].to_list()
    joint2id = {v: i for i, v in enumerate(sorted(set(joint_labels)))}
    label_ids = np.array([joint2id[v] for v in joint_labels])
    class_counts = np.bincount(label_ids, minlength=len(joint2id))
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    sample_weights = class_weights[label_ids]
    return class_weights, sample_weights


def compute_ce_class_weights(df: pl.DataFrame, use_viewpoint: bool = False):
    """Per-task inverse-frequency class weights for weighted cross-entropy.
    Each returned tensor is ordered by class id (SCENE2ID / VIEWPOINT2ID) so it
    can be passed straight to `nn.CrossEntropyLoss(weight=...)`."""

    def _weights(values, mapping):
        label_ids = np.array([mapping[v] for v in values])
        class_counts = np.bincount(label_ids, minlength=len(mapping))
        return class_counts.sum() / (len(mapping) * class_counts)

    weights = {"scene": _weights(df["label"].to_list(), SCENE2ID)}
    if use_viewpoint:
        weights["viewpoint"] = _weights(
            df["viewpoint_type"].to_list(), VIEWPOINT2ID
        )
    return weights


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
    use_viewpoint: bool = False,
):
    transform_fn = _build_transform(train=split == "train")

    dataset = ImageDataset(df, base_img_path, transform_fn, use_viewpoint=use_viewpoint)
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
