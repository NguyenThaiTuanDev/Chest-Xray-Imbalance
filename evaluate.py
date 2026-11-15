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
            if len(others) != 1:
                raise ValueError(f"{csv_path}: cannot unambiguously determine image ID column: {others}")
            id_candidates = others
        self.ids = frame[id_candidates[0]].astype(str).tolist()
        values = frame[labels].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
            raise ValueError(f"{csv_path}: labels must be nonmissing binary 0/1")
        self.targets = values
        # Validation is a subset of original training images, not a separate val image directory.
        image_dir = root / ("test" if split == "test" else "train")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        paths = []
        for image_id in self.ids:
            image_id_path = Path(image_id)
            candidates = [image_dir / image_id, root / image_id]
            if not image_id_path.suffix:
                candidates += [image_dir / f"{image_id}{ext}" for ext in IMAGE_EXTENSIONS]
            located = next((p for p in candidates if p.is_file()), None)
            if located is None:
                raise FileNotFoundError(f"Image '{image_id}' not found under {image_dir} (or {root})")
            if located.suffix.lower() == ".dcm":
                raise ValueError(f"DICOM image {located} requires an explicit DICOM preprocessing pipeline matching training.")
            paths.append(located)
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as img:
            # Typical ImageNet-pretrained DenseNet expects RGB (3 channels).
            x = self.transform(img.convert("RGB"))
        return x, torch.from_numpy(self.targets[index].copy())


def checkpoint_seed_dirs(root, dataset, seed):
    base = root / dataset
    return [p for p in (base / f"seed{seed}", base / f"seed_{seed}") if p.is_dir()]


def canonical_model_name(path: Path, seed_dir: Path):
    rel = path.relative_to(seed_dir)
    if len(rel.parts) > 1:
        # Prefer model/loss subdirectory, e.g. seed42/asl/best_checkpoint.pth
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", rel.parts[0]).lower()
    name = path.stem.lower()
    name = re.sub(r"(^|[_-])(best|checkpoint|ckpt|model|weights|state_dict|latest|last)(?=[_-]|$)", "_", name)
    name = re.sub(r"[_-]+", "_", name).strip("_")
    return name or "unknown_model"


def find_checkpoints(root: Path, dataset: str, seed: int, filename_pattern: str):
    """One preferred (best) file per model within a given dataset/seed."""
    import fnmatch
    selected = {}
    for seed_dir in checkpoint_seed_dirs(root, dataset, seed):
        candidates = sorted(p for p in seed_dir.rglob("*") if p.is_file()
                            and p.suffix.lower() in CHECKPOINT_EXTENSIONS
                            and fnmatch.fnmatch(p.name, filename_pattern)
                            and not any(t in p.stem.lower() for t in AVOID_TOKENS))
        for p in candidates:
            name = canonical_model_name(p, seed_dir)
            score = (int(any(t in p.stem.lower() for t in BEST_TOKENS)),
                     int(p.stem.lower() in ("best_checkpoint", "best_model", "best")))
            if name not in selected or score > selected[name][0]:
                selected[name] = (score, p)
    return [(name, item[1]) for name, item in sorted(selected.items())]


def load_checkpoint(path: Path):
    # weights_only=True rejects arbitrary Python objects in untrusted checkpoint files.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Could not safely read {path}. Only load trusted state_dict checkpoints. Details: {e}") from e
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint {path} does not contain a dict")
    for key in ("model_state_dict", "state_dict", "model_state", "model", "net", "network", "weights"):
        if key in payload and isinstance(payload[key], dict):
            state = payload[key]
            break
    else:
        if all(isinstance(v, torch.Tensor) for v in payload.values()):
            state = payload
        else:
            raise ValueError(f"Checkpoint {path}: no recognized model state_dict key; available keys {list(payload)[:25]}")
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"Checkpoint {path}: state_dict must map parameter names to tensors")
    return payload, state


def normalize_state_dict(state, expected):
    """Convert common DenseNet201 wrappers to torchvision native keys; never load partial weights."""
    mappings = []
    for original_key, tensor in state.items():
        key = original_key
        while key.startswith(("module.", "model.", "net.", "network.")):
            key = key.split(".", 1)[1]
        # The Stage-1 code may store DenseNet as backbone.features + a separate classifier.
        for prefix in ("backbone.", "encoder.", "encoder_q.", "densenet."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        if key.startswith("fc.") or key.startswith("head.") or key.startswith("classification_head."):
            key = "classifier." + key.split(".", 1)[1]
        mappings.append((key, tensor, original_key))
    actual = {}
    for key, tensor, source in mappings:
        if key not in expected:
            continue  # optimizer, projection head, unrelated buffers (never silently skip model params)
        if key in actual:
            raise ValueError(f"Duplicate mapped model parameter: {key}")
        if expected[key].shape != tensor.shape:
            raise ValueError(f"Shape mismatch {source} -> {key}: {tuple(tensor.shape)} vs {tuple(expected[key].shape)}")
        actual[key] = tensor
    missing = [k for k in expected if k not in actual and not k.endswith("num_batches_tracked")]
    if missing:
        raise ValueError(f"Checkpoint does not fully match DenseNet201. Missing {len(missing)} keys; examples: {missing[:12]}. "
                         "Inspect your training model class and adapt model-building/loading logic; evaluation has NOT been performed.")
    return actual


def build_model(path, device):
    payload, state = load_checkpoint(path)
    try:
        model = models.densenet201(weights=None)
    except TypeError:
        model = models.densenet201(pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 14)
    mapped = normalize_state_dict(state, model.state_dict())
    incompat = model.load_state_dict(mapped, strict=False)
    if incompat.missing_keys and any(not k.endswith("num_batches_tracked") for k in incompat.missing_keys):
        raise ValueError(f"Missing trainable model keys: {incompat.missing_keys}")
    model.to(device).eval()
    return model, payload


@torch.inference_mode()
def predict(model, loader, device, label):
    ys, ps = [], []
    for images, targets in tqdm(loader, desc=label, leave=False):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True))
        if logits.ndim != 2 or logits.shape[1] != 14:
            raise ValueError(f"Expected 14 logits, got {tuple(logits.shape)}")
        ys.append(targets.numpy())
        ps.append(torch.sigmoid(logits.float()).cpu().numpy())
    if not ys:
        raise ValueError("Empty dataset")
    return np.concatenate(ys), np.concatenate(ps)


def validation_thresholds(y, p):
    thresholds = []
    for c in range(y.shape[1]):
        yt, yp = y[:, c], p[:, c]
        if len(np.unique(yt)) < 2:
            thresholds.append(0.5)
            continue
        precision, recall, ts = precision_recall_curve(yt, yp)
        if not len(ts):
            thresholds.append(0.5)
            continue
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        thresholds.append(float(ts[int(np.nanargmax(f1))]))
    return np.asarray(thresholds, dtype=np.float64)


def get_saved_thresholds(payload, checkpoint_path, labels):
    for key in ("thresholds", "best_thresholds", "optimal_thresholds", "val_thresholds"):
        if key in payload:
            obj = payload[key]
            if isinstance(obj, torch.Tensor):
                obj = obj.detach().cpu().numpy()
            if isinstance(obj, dict):
                if set(obj) != set(labels):
                    raise ValueError(f"Stored {key} label names differ from configured label order")
                obj = [obj[name] for name in labels]
            result = np.asarray(obj, dtype=float)
            if result.shape == (len(labels),) and np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all():
                return result, f"checkpoint:{key}"
            raise ValueError(f"Invalid {key} shape/values in checkpoint")
    for filename in ("val_thresholds.json", "thresholds.json", "optimal_thresholds.json"):
        file = checkpoint_path.parent / filename
        if file.is_file():
            obj = json.loads(file.read_text(encoding="utf8"))
            if isinstance(obj, dict) and "thresholds" in obj:
                obj = obj["thresholds"]
            if isinstance(obj, dict):
                if set(obj) != set(labels):
                    raise ValueError(f"{file}: label names not identical to configured order")
                obj = [obj[k] for k in labels]
            result = np.asarray(obj, dtype=float)
            if result.shape != (len(labels),) or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
                raise ValueError(f"Invalid thresholds in {file}")
            return result, str(file)
    return None, None


def evaluate_metrics(y, p, thresholds, labels):
    records = []
    for c, label in enumerate(labels):
        yi, pi = y[:, c].astype(int), p[:, c]
        n_pos = int(yi.sum())
        valid = 0 < n_pos < len(yi)
        ap = float(average_precision_score(yi, pi)) if valid else float("nan")
        auc = float(roc_auc_score(yi, pi)) if valid else float("nan")
        pred = (pi >= thresholds[c]).astype(int)
        f1 = float(f1_score(yi, pred, zero_division=0)) if valid else float("nan")
        recall = float(np.sum((yi == 1) & (pred == 1)) / n_pos) if n_pos else float("nan")
        records.append({"class": label, "test_positives": n_pos, "test_negatives": int(len(yi) - n_pos),
                        "AP": ap, "ROC_AUC": auc, "F1": f1, "Recall": recall,
                        "threshold": float(thresholds[c])})
    df = pd.DataFrame(records)
    summary = {"mAP": float(df["AP"].mean()), "macro_AUC": float(df["ROC_AUC"].mean()),
               "macro_F1": float(df["F1"].mean()), "macro_Recall": float(df["Recall"].mean()),
               "valid_AP_classes": int(df["AP"].count()), "test_images": int(len(y))}
    return summary, df


def finite_json(value):
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    labels_by_dataset = load_labels(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA unavailable")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Image size: {args.image_size} | Checkpoints: {args.checkpoints_root}")
    all_rows, errors = [], []
    for dataset in args.datasets:
        labels = labels_by_dataset[dataset]
        data_dir = args.data_root / dataset
        runs = [(seed, model_name, path)
                for seed in args.seeds
                for model_name, path in find_checkpoints(args.checkpoints_root, dataset, seed, args.checkpoint_pattern)]
        if not runs:
            msg = f"No eligible checkpoint in {args.checkpoints_root / dataset} for requested seeds"
            if args.strict_coverage:
                raise FileNotFoundError(msg)
            print(f"WARNING: {msg}")
            continue
        print(f"\n{dataset}: found {len(runs)} checkpoint(s)")
        validation_set = XrayCSV(data_dir, "val", labels, args.image_size)
        test_set = XrayCSV(data_dir, "test", labels, args.image_size)
        loader_options = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda")
        val_loader = DataLoader(validation_set, **loader_options)
        test_loader = DataLoader(test_set, **loader_options)
        for seed, model_name, path in runs:
            print(f"  [{dataset} | seed {seed} | {model_name}] {path}")
            try:
                model, payload = build_model(path, device)
                thresholds, threshold_source = get_saved_thresholds(payload, path, labels)
                if thresholds is None:
                    y_val, p_val = predict(model, val_loader, device, "validation")
                    thresholds = validation_thresholds(y_val, p_val)
                    threshold_source = "computed_from_validation"
                y_test, p_test = predict(model, test_loader, device, "test")
                summary, class_table = evaluate_metrics(y_test, p_test, thresholds, labels)
                dest = args.output_dir / dataset / model_name / f"seed{seed}"
                dest.mkdir(parents=True, exist_ok=True)
                class_table.to_csv(dest / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
                save_json(dest / "thresholds_used.json", dict(zip(labels, thresholds.tolist())))
                np.savez_compressed(dest / "test_predictions.npz", y_true=y_test, y_prob=p_test,
                                    labels=np.asarray(labels, dtype=str))
                row = {"dataset": dataset, "model": model_name, "seed": seed, "checkpoint": str(path),
                       "threshold_source": threshold_source, **summary}
                save_json(dest / "test_metrics.json", row)
                all_rows.append(row)
                print(f"    mAP={summary['mAP']:.4f}, AUC={summary['macro_AUC']:.4f}, F1={summary['macro_F1']:.4f}")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                errors.append({"dataset": dataset, "seed": seed, "model": model_name,
                               "checkpoint": str(path), "error": str(exc)})
                print(f"    ERROR: {exc}", file=sys.stderr)
                if args.fail_fast:
                    raise
    if all_rows:
        runs_df = pd.DataFrame(all_rows).sort_values(["dataset", "model", "seed"])
        runs_df.to_csv(args.output_dir / "all_runs.csv", index=False, encoding="utf-8-sig")
        metric_cols = ["mAP", "macro_AUC", "macro_F1", "macro_Recall"]
        grouped = runs_df.groupby(["dataset", "model"], dropna=False)[metric_cols].agg(["mean", "std"])
        grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
        grouped = grouped.reset_index()
        grouped["num_seeds"] = runs_df.groupby(["dataset", "model"]).size().values
        grouped.sort_values(["dataset", "mAP_mean"], ascending=[True, False]).to_csv(
            args.output_dir / "summary_mean_std.csv", index=False, encoding="utf-8-sig")
        print(f"\nSaved {len(all_rows)} successful evaluations to {args.output_dir}")
        print(grouped[["dataset", "model", "num_seeds", "mAP_mean", "mAP_std", "macro_F1_mean"]].to_string(index=False))
    if errors:
        save_json(args.output_dir / "errors.json", errors)
        print(f"\n{len(errors)} run(s) failed. See {args.output_dir / 'errors.json'}", file=sys.stderr)
    if not all_rows:
        raise SystemExit("No checkpoints were successfully evaluated. Confirm architecture, data paths, CSV schema and checkpoint names.")
    if errors or (args.strict_coverage and any(not find_checkpoints(args.checkpoints_root, d, s, args.checkpoint_pattern)
                                                for d in args.datasets for s in args.seeds)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
