from typing import Literal

import numpy as np
import torch
import polars as pl
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
import os
from tqdm.auto import tqdm

from torch import nn
from src.viewpoint.model import ViewpointClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support


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


def compute_class_weights(df: pl.DataFrame):
    label_ids = np.array([LABEL2ID[v] for v in df["viewpoint"].to_list()])
    class_counts = np.bincount(label_ids, minlength=len(LABEL2ID))
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    sample_weights = class_weights[label_ids]
    return class_weights, sample_weights


def make_dataloader(
    df: pl.DataFrame,
    base_img_path: str,
    batch_size: int,
    split: Literal["train", "val", "test"],
    sampler=None,
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
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=4
    )


def train_epoch(model, criterion, optimizer, loaders, device):
    loss_stat = {"train": 0, "val": 0}
    acc_stat = {"train": 0, "val": 0}
    f1_stat = {"train": 0, "val": 0}
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0
        running_correct = 0
        n_samples = 0
        all_preds = []
        all_labels = []

        for inp, lbl in tqdm(
            loaders[phase], leave=False, desc=f"Running {phase} phase.."
        ):
            inp, lbl = inp.to(device), lbl.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                out = model(inp)
                preds = torch.argmax(out, dim=1)
                loss = criterion(out, lbl)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item()
            running_correct += (preds == lbl).sum().item()
            n_samples += lbl.size(0)
            all_preds.append(preds.detach().cpu())
            all_labels.append(lbl.detach().cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        epoch_loss = running_loss / len(loaders[phase])
        epoch_acc = running_correct / n_samples
        _, _, epoch_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        print(
            f"{phase.capitalize()} Loss: {epoch_loss:.4f} | "
            f"{phase.capitalize()} Acc: {epoch_acc:.4f} | "
            f"{phase.capitalize()} Macro-F1: {epoch_f1:.4f}"
        )
        loss_stat[phase] = epoch_loss
        acc_stat[phase] = epoch_acc
        f1_stat[phase] = epoch_f1

    return loss_stat, acc_stat, f1_stat


@torch.no_grad()
def run_inference(model, criterion, loader, device):
    model.eval()

    running_loss = 0
    running_correct = 0
    n_samples = 0
    all_preds = []
    all_labels = []

    for inp, lbl in tqdm(loader, leave=False, desc="Running inference.."):
        inp, lbl = inp.to(device), lbl.to(device)

        out = model(inp)
        preds = torch.argmax(out, dim=1)
        loss = criterion(out, lbl)

        running_loss += loss.item()
        running_correct += (preds == lbl).sum().item()
        n_samples += lbl.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(lbl.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    test_loss = running_loss / len(loader)
    test_acc = running_correct / n_samples
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=LABEL2ID["exterior_facing_landmark"]
    )
    print(
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
    )
    return {
        "loss": test_loss,
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


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
    _, sample_weights = compute_class_weights(train)
    train_sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    trainloader = make_dataloader(
        train, args.base_img_path, args.batch_size, split="train", sampler=train_sampler
    )
    valloader = make_dataloader(val, args.base_img_path, args.batch_size, split="val")
    testloader = make_dataloader(
        test, args.base_img_path, args.batch_size, split="test"
    )

    ### MODEL ###
    model = ViewpointClassifier(nclass=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)

    ### TRAIN ###
    best_f1 = -np.inf
    for ep in range(args.epochs):
        print(f"Epoch {ep+1}/{args.epochs}")
        _, _, f1s = train_epoch(
            model,
            criterion,
            optimizer,
            loaders={"train": trainloader, "val": valloader},
            device=device
        )

        if f1s["val"] > best_f1:
            best_f1 = f1s["val"]
            torch.save(
                model.state_dict(), os.path.join(args.ckpt_dir, "viewpoint_best.pt")
            )
        print("-" * 10)

    ### INFERENCE ###
    print("running inference on test set with best checkpoint")
    model.load_state_dict(
        torch.load(os.path.join(args.ckpt_dir, "viewpoint_best.pt"), map_location=device)
    )
    run_inference(model, criterion, testloader, device)


def infer(args):
    device = (
        "mps"
        if torch.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    ### DATASET ###
    print("split data and making test dataloader")
    df = pl.read_csv(args.data_path)
    _, test = train_test_split(
        df, test_size=0.05, random_state=42, stratify=df["viewpoint"]
    )
    testloader = make_dataloader(
        test, args.base_img_path, args.batch_size, split="test"
    )

    ### MODEL ###
    model = ViewpointClassifier(nclass=2).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    ### INFERENCE ###
    run_inference(model, criterion, testloader, device)


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-path", required=True)
    train_parser.add_argument(
        "--base-img-path", default="/home/affahrizain/projects/datasets/geotir"
    )
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--ckpt-dir", default="checkpoints/viewpoint/")

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--data-path", required=True)
    infer_parser.add_argument(
        "--base-img-path", default="/home/affahrizain/projects/datasets/geotir"
    )
    infer_parser.add_argument("--batch-size", type=int, default=128)
    infer_parser.add_argument("--ckpt-path", required=True)

    args = parser.parse_args()

    if args.command == "train":
        main(args)
    else:
        infer(args)