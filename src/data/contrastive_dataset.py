"""Two-view dataset for momentum contrastive training.

Uses exact label ordering and image decoding from baseline VinDrDataset.
Each view is sampled independently from the same train transform.
Validation/test must still use the baseline single-view VinDrDataset.
"""
from pathlib import Path

from src.data.dataset import VinDrDataset
from src.data.transforms import get_transforms


class ContrastiveVinDrDataset(VinDrDataset):
    def __init__(self, data_dir, dataset_name: str, image_size: int = 512,
                 strict_images: bool = True):
        data_dir = Path(data_dir)
        if dataset_name not in ("VinDr-CXR", "VinDr-PCXR"):
            raise ValueError("Unsupported VinDr dataset")
        if image_size < 32:
            raise ValueError("image_size must be >= 32")
        super().__init__(
            csv_file=data_dir / "labels" / "train_split.csv",
            img_dir=data_dir / "train",
            dataset_name=dataset_name,
            transform=get_transforms("train", image_size=image_size),
            is_contrastive=True,
            strict_binary=True,
            strict_images=strict_images,
        )
