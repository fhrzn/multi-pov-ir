import numpy as np
import torch
import polars as pl
from argparse import ArgumentParser
from torch.utils.data import WeightedRandomSampler
import os
from tqdm.auto import tqdm

from torch import nn
from src.viewpoint.data import compute_class_weights, make_dataloader
from src.viewpoint.model import ViewpointClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support


def train_epoch(model, criterion, optimizer, loaders, device):
    tasks = ["scene", "viewpoint"]
    loss_stat = {"train": 0, "val": 0}
    acc_stat = {"train": {t: 0 for t in tasks}, "val": {t: 0 for t in tasks}}
    f1_stat = {"train": {t: 0 for t in tasks}, "val": {t: 0 for t in tasks}}
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0
        running_correct = {t: 0 for t in tasks}
        n_samples = 0
        all_preds = {t: [] for t in tasks}
        all_labels = {t: [] for t in tasks}

        for inp, scene_lbl, viewpoint_lbl in tqdm(
            loaders[phase], leave=False, desc=f"Running {phase} phase.."
        ):
            inp = inp.to(device)
            scene_lbl, viewpoint_lbl = scene_lbl.to(device), viewpoint_lbl.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                scene_out, viewpoint_out = model(inp)
                scene_preds = torch.argmax(scene_out, dim=1)
                viewpoint_preds = torch.argmax(viewpoint_out, dim=1)
                loss = criterion(scene_out, scene_lbl) + criterion(viewpoint_out, viewpoint_lbl)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item()
            running_correct["scene"] += (scene_preds == scene_lbl).sum().item()
            running_correct["viewpoint"] += (viewpoint_preds == viewpoint_lbl).sum().item()
            n_samples += scene_lbl.size(0)
            all_preds["scene"].append(scene_preds.detach().cpu())
            all_labels["scene"].append(scene_lbl.detach().cpu())
            all_preds["viewpoint"].append(viewpoint_preds.detach().cpu())
            all_labels["viewpoint"].append(viewpoint_lbl.detach().cpu())

        epoch_loss = running_loss / len(loaders[phase])
        loss_stat[phase] = epoch_loss
        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")

        for t in tasks:
            preds = torch.cat(all_preds[t]).numpy()
            labels = torch.cat(all_labels[t]).numpy()
            epoch_acc = running_correct[t] / n_samples
            _, _, epoch_f1, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0
            )
            print(
                f"  [{t}] {phase.capitalize()} Acc: {epoch_acc:.4f} | "
                f"{phase.capitalize()} Macro-F1: {epoch_f1:.4f}"
            )
            acc_stat[phase][t] = epoch_acc
            f1_stat[phase][t] = epoch_f1

    return loss_stat, acc_stat, f1_stat


@torch.no_grad()
def run_evaluate(model, criterion, loader, device):
    model.eval()

    tasks = ["scene", "viewpoint"]
    running_loss = 0
    running_correct = {t: 0 for t in tasks}
    n_samples = 0
    all_preds = {t: [] for t in tasks}
    all_labels = {t: [] for t in tasks}

    for inp, scene_lbl, viewpoint_lbl in tqdm(loader, leave=False, desc="Running evaluation.."):
        inp = inp.to(device)
        scene_lbl, viewpoint_lbl = scene_lbl.to(device), viewpoint_lbl.to(device)

        scene_out, viewpoint_out = model(inp)
        scene_preds = torch.argmax(scene_out, dim=1)
        viewpoint_preds = torch.argmax(viewpoint_out, dim=1)
        loss = criterion(scene_out, scene_lbl) + criterion(viewpoint_out, viewpoint_lbl)

        running_loss += loss.item()
        running_correct["scene"] += (scene_preds == scene_lbl).sum().item()
        running_correct["viewpoint"] += (viewpoint_preds == viewpoint_lbl).sum().item()
        n_samples += scene_lbl.size(0)
        all_preds["scene"].append(scene_preds.cpu())
        all_labels["scene"].append(scene_lbl.cpu())
        all_preds["viewpoint"].append(viewpoint_preds.cpu())
        all_labels["viewpoint"].append(viewpoint_lbl.cpu())

    test_loss = running_loss / len(loader)
    print(f"Test Loss: {test_loss:.4f}")

    results = {"loss": test_loss}
    for t in tasks:
        preds = torch.cat(all_preds[t]).numpy()
        labels = torch.cat(all_labels[t]).numpy()
        test_acc = running_correct[t] / n_samples
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        print(
            f"  [{t}] Acc: {test_acc:.4f} | Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | F1: {f1:.4f}"
        )
        results[t] = {
            "accuracy": test_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return results


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
    strat_key = df["scene_type"] + "_" + df["viewpoint_type"]
    train, test = train_test_split(
        df, test_size=0.05, random_state=42, stratify=strat_key
    )
    train_strat_key = train["scene_type"] + "_" + train["viewpoint_type"]
    train, val = train_test_split(
        train, test_size=0.2, random_state=42, stratify=train_strat_key
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
    model = ViewpointClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    head_params = list(model.scene_head.parameters()) + list(model.viewpoint_head.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=args.lr)

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

        val_f1 = (f1s["val"]["scene"] + f1s["val"]["viewpoint"]) / 2
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                model.state_dict(), os.path.join(args.ckpt_dir, "viewpoint_best.pt")
            )
        print("-" * 10)

    ### INFERENCE ###
    print("running inference on test set with best checkpoint")
    model.load_state_dict(
        torch.load(os.path.join(args.ckpt_dir, "viewpoint_best.pt"), map_location=device)
    )
    run_evaluate(model, criterion, testloader, device)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--base-img-path", default="/home/affahrizain/projects/datasets/geotir"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt-dir", default="checkpoints/viewpoint/")

    args = parser.parse_args()
    main(args)