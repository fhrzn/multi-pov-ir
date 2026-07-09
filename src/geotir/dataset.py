import os

import polars as pl
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class LandmarkDataset(Dataset):
    def __init__(
        self,
        processor: CLIPProcessor,
        df: pl.DataFrame,
        base_img_path: str,
        max_length: int = 77,
    ):
        super().__init__()

        self.df = df
        self.base_img_path = base_img_path
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        image = Image.open(
            os.path.join(
                self.base_img_path, row["src"], row["id"] + ".jpg"
            )
        ).convert("RGB")

        processed = self.processor(
            images=image,
            text=row["caption"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "category": row["category"],
            "country": row["country"],
            **{k: v.squeeze(0) for k, v in processed.items()},
        }
