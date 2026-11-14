from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import CLASSES, VinDrDataset
from src.data.transforms import get_transforms
from src.models.builder import build_model


# 1. OUTPUT COLUMNS
SUMMARY_COLUMNS = [
    "Seed_Folder",
    "Checkpoint",
    "Macro_ROC_AUC",
    "Macro_PR_AUC_mAP",
    "Macro_F1_OptThresh",
]

# 2. OPTIMAL THRESHOLD ON VALIDATION
def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the threshold maximizing F1 on validation.
    For a class containing only positives or only negatives,
    use a fallback threshold of 0.5. Never optimize on test data.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_true.ndim != 1 or y_true.shape != y_prob.shape:
        raise ValueError("Expected matching one-dimensional arrays")

    if np.unique(y_true).size < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5

    denominator = np.maximum(precision[:-1] + recall[:-1], 1e-12)
    f1_values = 2.0 * precision[:-1] * recall[:-1] / denominator
    best_index = int(np.argmax(f1_values))

    return float(thresholds[best_index])

# 3. INFERENCE WITH TQDM
@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    progress_label: str = "Inference",
):
    """Run inference and return y_true [N, C] and y_prob [N, C]."""
    model.eval()
    true_batches, prob_batches = [], []

    progress_bar = tqdm(
        loader,
        desc=progress_label,
        unit="batch",
        leave=False,
        position=1,
        dynamic_ncols=True,
        mininterval=0.5,
    )

    for images, targets in progress_bar:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.sigmoid(logits.float())

        true_batches.append(targets.detach().cpu().numpy())
        prob_batches.append(probabilities.cpu().numpy())

    if not true_batches:
        raise ValueError("Dataset is empty. Check CSV and image paths.")

    y_true = np.concatenate(true_batches, axis=0)
    y_prob = np.concatenate(prob_batches, axis=0)

    if y_true.ndim != 2 or y_true.shape != y_prob.shape:
        raise ValueError(f"Prediction shape mismatch: {y_true.shape} vs {y_prob.shape}")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("Dataset labels must be binary (0/1)")
    if not np.isfinite(y_prob).all():
        raise ValueError("Predicted probabilities contain NaN or Inf")

    return y_true, y_prob

# 4. CREATE VALIDATION / TEST DATALOADERS
def make_loader(
    data_dir: Path,
    split: str,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    if split not in ("val", "test"):
        raise ValueError("split must be either 'val' or 'test'")

    csv_name = "val_split.csv" if split == "val" else "test_labels.csv"
    image_subdir = "train" if split == "val" else "test"

    csv_file = data_dir / "labels" / csv_name
    image_dir = data_dir / image_subdir

    if not csv_file.is_file():
        raise FileNotFoundError(f"Missing {split} CSV: {csv_file}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing {split} images: {image_dir}")

    dataset = VinDrDataset(
        csv_file=str(csv_file),
        img_dir=str(image_dir),
        dataset_name=dataset_name,
        transform=get_transforms("test"),
    )

    if list(dataset.classes) != list(CLASSES[dataset_name]):
        raise ValueError("Dataset class order does not match CLASSES")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# 5. LOAD CHECKPOINT
def load_state_dict(checkpoint_path: Path) -> dict:
    """Load and clean state_dict from checkpoint."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint: {checkpoint_path}")

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )

    if not isinstance(state_dict, dict):
        raise ValueError(f"Missing state_dict: {checkpoint_path}")
    if not state_dict:
        raise ValueError(f"Empty state_dict: {checkpoint_path}")

    cleaned_state_dict = {
        str(key).removeprefix("module."): value
        for key, value in state_dict.items()
    }

    if not all(torch.is_tensor(v) for v in cleaned_state_dict.values()):
        raise ValueError(f"Invalid model weights: {checkpoint_path}")

    return cleaned_state_dict

# 6. PARSE CHECKPOINT INFORMATION
def parse_checkpoint(checkpoint_path: Path):
    """Extract (seed, loss_config, seed_folder) from e.g. best_bce_seed42.pth."""
    match = re.fullmatch(
        r"best_(.+)_seed_?(\d+)\.pth",
        checkpoint_path.name,
        flags=re.I,
    )
    if match is None:
        raise ValueError(f"Unexpected checkpoint name: {checkpoint_path.name}")

    loss_config = match.group(1).lower()
    seed = int(match.group(2))
    seed_folder = checkpoint_path.parent.name

    folder_match = re.fullmatch(r"seed_?(\d+)", seed_folder, flags=re.I)
    if folder_match is not None and int(folder_match.group(1)) != seed:
        raise ValueError(f"Checkpoint seed mismatch: {checkpoint_path}")

    return seed, loss_config, seed_folder

# 7. COMPUTE METRICS
def score_checkpoint(val_y, val_prob, test_y, test_prob, label_names):
    """Calculate test metrics using thresholds optimized on validation."""
    n_classes = len(label_names)
    if val_y.shape[1] != n_classes or test_y.shape[1] != n_classes:
        raise ValueError("Number of labels does not match model outputs")

    thresholds = np.array(
        [optimal_threshold(val_y[:, j], val_prob[:, j]) for j in range(n_classes)],
        dtype=float,
    )

    val_valid = (val_y.sum(axis=0) > 0) & (val_y.sum(axis=0) < len(val_y))
    test_valid = (test_y.sum(axis=0) > 0) & (test_y.sum(axis=0) < len(test_y))

    auc_scores, ap_scores, f1_scores = [], [], []
    per_class_rows = []

    for j, label in enumerate(label_names):
        auc = ap = f1 = float("nan")

        if test_valid[j]:
            auc = float(roc_auc_score(test_y[:, j], test_prob[:, j]))
            ap = float(average_precision_score(test_y[:, j], test_prob[:, j]))
            binary_preds = (test_prob[:, j] >= thresholds[j]).astype(np.uint8)
            f1 = float(f1_score(test_y[:, j], binary_preds, zero_division=0))

            auc_scores.append(auc)
            ap_scores.append(ap)
            f1_scores.append(f1)

        per_class_rows.append({
            "Label": label,
            "ValPos": int(val_y[:, j].sum()),
            "TestPos": int(test_y[:, j].sum()),
            "Val_Valid": bool(val_valid[j]),
            "Test_Valid": bool(test_valid[j]),
            "AP": ap,
            "AUC": auc,
            "F1_ValThresh": f1,
            "Threshold": float(thresholds[j]),
            "Threshold_Source": "validation",
        })

    metrics = {
        "Macro_ROC_AUC": float(np.mean(auc_scores)) if auc_scores else float("nan"),
        "Macro_PR_AUC_mAP": float(np.mean(ap_scores)) if ap_scores else float("nan"),
        "Macro_F1_OptThresh": float(np.mean(f1_scores)) if f1_scores else float("nan"),
    }

    return metrics, per_class_rows

# 8. MAIN EVALUATION PIPELINE
def evaluate_all(
    dataset_name: str,
    checkpoints_dir: str | None = None,
    batch_size: int = 1,
    num_workers: int = 2,
    output_dir: str = "../outputs/csv",
    data_root: str = "data/processed",
    device_name: str = "auto",
):
    if dataset_name not in CLASSES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    data_dir = Path(data_root) / dataset_name
    checkpoint_root = (
        Path(checkpoints_dir)
        if checkpoints_dir
        else Path("outputs/checkpoints") / dataset_name
    )

    checkpoint_paths = sorted(checkpoint_root.rglob("best_*_seed*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_root}")

    # Check duplicate seed/loss configurations.
    seen = set()
    for path in checkpoint_paths:
        seed, loss_config, _ = parse_checkpoint(path)
        key = (seed, loss_config)
        if key in seen:
            raise ValueError(f"Duplicate seed/config: {key}")
        seen.add(key)

    # Device selection.
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the GPU")

    use_cuda = device_name == "cuda" or (
        device_name == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")

    # Dataloaders & Model.
    val_loader = make_loader(data_dir, "val", dataset_name, batch_size, num_workers)
    test_loader = make_loader(data_dir, "test", dataset_name, batch_size, num_workers)

    model = build_model(
        model_name="densenet201",
        num_classes=len(CLASSES[dataset_name]),
        pretrained=False,
    ).to(device)

    tqdm.write(
        f"Dataset: {dataset_name} | "
        f"Device: {device} | "
        f"Checkpoints: {len(checkpoint_paths)}"
    )

    summary_rows, per_class_long, per_class_wide = [], [], []
    reference_val = reference_test = None

    # MAIN TQDM: CHECKPOINTS
    checkpoint_bar = tqdm(
        checkpoint_paths,
        desc=f"{dataset_name} checkpoints",
        unit="ckpt",
        position=0,
        dynamic_ncols=True,
    )

    for index, checkpoint_path in enumerate(checkpoint_bar, start=1):
        tqdm.write(f"\n[{index}/{len(checkpoint_paths)}] Loading {checkpoint_path.name}")

        seed, loss_config, seed_folder = parse_checkpoint(checkpoint_path)
        state_dict = load_state_dict(checkpoint_path)
        model.load_state_dict(state_dict, strict=True)

        val_y, val_prob = predict(model, val_loader, device, progress_label="Validation")
        test_y, test_prob = predict(model, test_loader, device, progress_label="Test")

        # Check dataset consistency.
        if reference_val is None:
            reference_val = val_y.copy()
            reference_test = test_y.copy()
        elif not (
            np.array_equal(reference_val, val_y)
            and np.array_equal(reference_test, test_y)
        ):
            raise ValueError("Dataset labels/order changed between checkpoint evaluations")

        # Compute metrics.
        metrics, class_rows = score_checkpoint(
            val_y=val_y,
            val_prob=val_prob,
            test_y=test_y,
            test_prob=test_prob,
            label_names=CLASSES[dataset_name],
        )

        # Main CSV (5 columns).
        summary_rows.append({
            "Seed_Folder": seed_folder,
            "Checkpoint": checkpoint_path.name,
            **metrics,
        })

        # Wide AP table.
        per_class_wide.append({
            "Seed_Folder": seed_folder,
            "Checkpoint": checkpoint_path.name,
            **{row["Label"]: row["AP"] for row in class_rows},
            "Macro_PR_AUC_mAP": metrics["Macro_PR_AUC_mAP"],
        })

        # Long per-class table.
        for row in class_rows:
            per_class_long.append({
                "Dataset": dataset_name,
                "Seed": seed,
                "Seed_Folder": seed_folder,
                "Loss_Config": loss_config,
                "Checkpoint": checkpoint_path.name,
                **row,
            })

        valid_count = sum(row["Test_Valid"] for row in class_rows)

        checkpoint_bar.set_postfix({
            "mAP": f"{metrics['Macro_PR_AUC_mAP']:.4f}",
            "AUC": f"{metrics['Macro_ROC_AUC']:.4f}",
            "F1": f"{metrics['Macro_F1_OptThresh']:.4f}",
        })

        tqdm.write(
            f"[{index}/{len(checkpoint_paths)}] {checkpoint_path.name} | "
            f"AUC={metrics['Macro_ROC_AUC']:.5f} | "
            f"mAP={metrics['Macro_PR_AUC_mAP']:.5f} | "
            f"F1={metrics['Macro_F1_OptThresh']:.5f} | "
            f"Valid={valid_count}/{len(class_rows)}"
        )
    # 9. SAVE CSV FILES
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_file = save_dir / f"evaluation_results_{dataset_name}_val_thresholds.csv"
    wide_file = save_dir / f"evaluation_per_class_AP_{dataset_name}_val_thresholds.csv"
    long_file = save_dir / f"evaluation_per_class_long_{dataset_name}_val_thresholds.csv"
    thresholds_file = save_dir / f"evaluation_thresholds_long_{dataset_name}_val_thresholds.csv"

    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(
        summary_file, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(per_class_wide).to_csv(
        wide_file, index=False, encoding="utf-8-sig"
    )

    df_long = pd.DataFrame(per_class_long)
    df_long.to_csv(long_file, index=False, encoding="utf-8-sig")

    threshold_columns = [
        "Dataset",
        "Seed",
        "Seed_Folder",
        "Loss_Config",
        "Checkpoint",
        "Label",
        "ValPos",
        "Threshold",
        "Threshold_Source",
    ]
    df_long[threshold_columns].to_csv(
        thresholds_file, index=False, encoding="utf-8-sig"
    )

    saved_files = [summary_file, wide_file, long_file, thresholds_file]
    for file_path in saved_files:
        tqdm.write(f"Saved: {file_path.resolve()}")

    return saved_files

# 10. COMMAND LINE ARGUMENTS
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DenseNet201 checkpoints with validation-only F1 thresholds and tqdm"
    )
    parser.add_argument("--dataset", choices=list(CLASSES), required=True)
    parser.add_argument("--checkpoints_dir", default=None)
    parser.add_argument("--data_root", default="data/processed")
    parser.add_argument("--output_dir", default="../outputs/csv")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    args = parser.parse_args()

    evaluate_all(
        dataset_name=args.dataset,
        checkpoints_dir=args.checkpoints_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        data_root=args.data_root,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()