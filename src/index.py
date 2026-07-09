import torch
from src.geotir.dataset import LandmarkDataset
from src.geotir.model import GeoTIRModel
from transformers import AutoProcessor
import polars as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
import randomname

from src.utils import get_device, build_index, add_record_to_index, save_index

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
INDEX_SIZE = 768

def _setup_geotir(args, device):
    if not args.ckpt_path:
        raise ValueError("--ckpt-path is required for --model geotir")
    model = GeoTIRModel(clip_model_name=CLIP_MODEL_NAME).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.eval()
    model = torch.compile(model)
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)

    def encode(batch):
        with torch.autocast(device, dtype=torch.bfloat16):
            out = model.encode_images(pixel_values=batch["pixel_values"].to(device))
        return out.cpu().float().numpy()

    return processor, encode

def ingest(args):
    device = get_device()
    processor, encode_fn =_setup_geotir(args, device)

    df = pl.read_csv(args.data_path)
    if "category" not in df.columns:
        try:
            df = df.rename({"pred_label": "category"})
        except Exception:
            df = df.rename({"predicted_label": "category"})

    dataset = LandmarkDataset(processor, df, args.img_base_path)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    index = build_index(INDEX_SIZE, args.index_type)

    with torch.no_grad():
        for batch in tqdm(loader, desc="encode"):
            embeddings = encode_fn(batch)
            add_record_to_index(index, embeddings)

    target_dir = f"index/{args.output_dir if args.output_dir else randomname.generate(sep='_')}"
    save_index(index, df.to_dicts(), target_dir=target_dir)
    print(f"index and metadata saved successfully to {target_dir}")

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--img-base-path", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--index-type", default="flat_ip")
    parser.add_argument("--output-dir")

    args = parser.parse_args()

    ingest(args)
