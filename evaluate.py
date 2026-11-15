from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

DEFAULT_LABELS = {
    "VinDr-CXR": [
        "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
        "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
        "Other lesion", "Pleural effusion", "Pleural thickening",
        "Pneumothorax", "Pulmonary fibrosis",
    ],
    "VinDr-PCXR": [
        "Bronchitis", "Brocho-pneumonia", "Other disease", "Bronchiolitis",
        "Situs inversus", "Pneumonia", "Pleuro-pneumonia", "Diagphramatic hernia",
        "Tuberculosis", "Congenital emphysema", "CPAM",
        "Hyaline membrane disease", "Mediastinal tumor", "Lung tumor",
    ],
}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".dcm")
ID_COLUMNS = ("image_id", "image_path", "filename", "path", "id")
CHECKPOINT_EXTENSIONS = (".pt", ".pth", ".ckpt")
BEST_TOKENS = ("best",)
AVOID_TOKENS = ("latest", "last", "final_epoch", "optimizer", "scheduler")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints"))
    p.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    p.add_argument("--datasets", nargs="+", default=["VinDr-CXR", "VinDr-PCXR"], choices=list(DEFAULT_LABELS))
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 132, 2024, 3407])
    p.add_argument("--labels-json", type=Path, help="JSON map: dataset name -> ordered list of 14 class names")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--checkpoint-pattern", default="*", help="Filter checkpoint filename using shell glob (e.g. '*best*.pth')")
    p.add_argument("--strict-coverage", action="store_true", help="Stop on incomplete dataset/seed coverage")
    p.add_argument("--fail-fast", action="store_true", help="Stop on first evaluation failure")
    return p.parse_args()


def load_labels(args):
    labels = {k: list(v) for k, v in DEFAULT_LABELS.items()}
    if args.labels_json:
        override = json.loads(args.labels_json.read_text(encoding="utf-8"))
        labels.update(override)
    for dataset in args.datasets:
        if dataset not in labels or len(labels[dataset]) != 14 or len(set(labels[dataset])) != 14:
            raise ValueError(f"{dataset}: need exactly 14 unique labels in correct trained order")
    return labels


class XrayCSV(Dataset):
    def __init__(self, root: Path, split: str, labels: list[str], size: int):
        self.root, self.split, self.labels = root, split, labels
        csv_path = root / "labels" / f"{split}_split.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing label CSV: {csv_path}")
        frame = pd.read_csv(csv_path)
        missing = [c for c in labels if c not in frame.columns]
        if missing:
            raise ValueError(f"{csv_path}: missing label columns: {missing}. Use --labels-json with exact training label names/order if different.")
        id_candidates = [c for c in ID_COLUMNS if c in frame.columns]
        if not id_candidates:
            others = [c for c in frame.columns if c not in labels]