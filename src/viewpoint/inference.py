from argparse import ArgumentParser

import polars as pl
import torch
from tqdm.auto import tqdm

from src.viewpoint.data import SCENE2ID, VIEWPOINT2ID, make_inference_dataloader
from src.viewpoint.model import ViewpointClassifier

ID2SCENE = {v: k for k, v in SCENE2ID.items()}
ID2VIEWPOINT = {v: k for k, v in VIEWPOINT2ID.items()}


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()

    tasks = ["scene", "viewpoint"]
    all_preds = {t: [] for t in tasks}

    for inp in tqdm(loader, leave=False, desc="Running inference.."):
        inp = inp.to(device)

        scene_out, viewpoint_out = model(inp)
        all_preds["scene"].append(torch.argmax(scene_out, dim=1).cpu())
        all_preds["viewpoint"].append(torch.argmax(viewpoint_out, dim=1).cpu())

    return {
        "scene": torch.cat(all_preds["scene"]).numpy(),
        "viewpoint": torch.cat(all_preds["viewpoint"]).numpy(),
    }


def main(args):
    device = (
        "mps"
        if torch.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    ### DATASET ###
    df = pl.read_csv(args.input)
    loader = make_inference_dataloader(df, args.base_img_path, args.batch_size)

    ### MODEL ###
    model = ViewpointClassifier().to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))

    ### INFERENCE ###
    preds = run_inference(model, loader, device)

    out_df = df.with_columns(
        pl.Series("scene_pred", [ID2SCENE[p] for p in preds["scene"]]),
        pl.Series("viewpoint_pred", [ID2VIEWPOINT[p] for p in preds["viewpoint"]]),
    )
    out_df.write_csv(args.output)
    print(f"saved predictions to {args.output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument(
        "--base-img-path", default="/home/affahrizain/projects/datasets/geotir"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("-o", "--output", default="predictions.csv")

    args = parser.parse_args()
    main(args)
