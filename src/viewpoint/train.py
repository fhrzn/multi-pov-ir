from typing import Literal

import numpy as np
import torch
import polars as pl
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
from tqdm.auto import tqdm

from torch import nn
from src.viewpoint.model import ViewpointClassifier
from sklearn.model_selection import train_test_split


LABEL2ID = {"interior": 0, "exterior_facing_landmark": 1}


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
        label = LABEL2ID[row["viewpoint"]]
        if self.transform:
            image = self.transform(image)
        return image, label


def make_dataloader(
    df: pl.DataFrame,
    base_img_path: str,
    batch_size: int,
    split: Literal["train", "val", "test"],
):
    if split == "train":
        transform_fn = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform_fn = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    dataset = ImageDataset(df, base_img_path, transform_fn)
    return DataLoader(
        dataset,
        batch_size,
        shuffle=True,
    )


def train_epoch(model, criterion, optimizer, loaders, device):
    loss_stat = {"train": 0, "val": 0}
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        loss = 0

        for inp, lbl in tqdm(
            loaders[phase], leave=True, desc=f"Running {phase} phase.."
        ):
            inp, lbl = inp.to(device), lbl.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                out = model(inp)
                # _, preds = torch.max(out, 1)
                loss = criterion(out, lbl)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            loss += loss.item()

        print(f"{phase.capitalize()} Loss: {loss / len(loaders):.2f}")
        loss_stat[phase] = loss / len(loaders)

    return loss_stat


def main(args):
    ### CONFIG ###
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = (
        "mps"
        if torch.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    ### DATASET ###
    print("split data and making dataloaders")
    df = pl.read_csv(args.data_path)
    train, test = train_test_split(
        df, test_size=0.05, random_state=42, stratify=df["viewpoint"]
    )
    train, val = train_test_split(
        train, test_size=0.2, random_state=42, stratify=train["viewpoint"]
    )
    trainloader = make_dataloader(
        train, args.base_img_path, args.batch_size, split="train"
    )
    valloader = make_dataloader(val, args.base_img_path, args.batch_size, split="val")
    testloader = make_dataloader(
        test, args.base_img_path, args.batch_size, split="test"
    )

    ### MODEL ###
    model = ViewpointClassifier(nclass=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters())

    ### TRAIN ###
    best_loss = np.inf
    for ep in range(args.epochs):
        print(f"Epoch {ep+1}/{args.epochs}")
        print("-" * 10)
        losses = train_epoch(
            model,
            criterion,
            optimizer,
            loaders={"train": trainloader, "val": valloader},
            device=device
        )

        if losses["val"] < best_loss:
            best_loss = losses["val"]
            torch.save(
                model.state_dict(), os.path.join(args.ckpt_dir, "viewpoint_best.pt")
            )

    ### INFERENCE ###


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--base-img-path", default="/home/affahrizain/projects/datasets/geotir"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ckpt-dir", default="checkpoints/viewpoint/")
    
    args = parser.parse_args()

    main(args)