from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--checkpoints-root", type=Path, default=Path("outputs/stage1/checkpoints"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/stage1"), help="Stage-1 output root containing csv/ and visualizations/")
    p.add_argument("--flow", choices=["main", "detail", "both"], default="both", help="main: major metrics; detail: per-label metrics + figures; both: one inference pass")
    p.add_argument("--output-prefix", default="eval_", help="Prefix for generated CSV/PNG names; default avoids overwriting prior Stage-1 reports")
    p.add_argument("--expected-models", type=int, default=14, help="Expected Stage-1 loss configurations per dataset/seed for strict coverage")
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
        label_dir = root / "labels"
        # Different preprocessing notebooks export test_labels.csv rather than test_split.csv.
        # Prefer the explicitly named labels file for test, while supporting either convention.
        candidates = ([label_dir / "test_labels.csv", label_dir / "test_split.csv"]
                      if split == "test" else
                      [label_dir / f"{split}_split.csv", label_dir / f"{split}_labels.csv"])
        csv_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if csv_path is None:
            available = sorted(p.name for p in label_dir.glob("*.csv")) if label_dir.is_dir() else []
            raise FileNotFoundError(
                f"Missing {split} labels under {label_dir}. "
                f"Looked for {[str(p) for p in candidates]}. "
                f"Available CSVs: {available}"
            )
        print(f"[DATA] {root.name}/{split}: {csv_path}", flush=True)
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
        id_column = id_candidates[0]
        if frame[id_column].isna().any():
            raise ValueError(f"{csv_path}: missing image IDs in column {id_column}")
        self.ids = frame[id_column].astype(str).tolist()
        if len(self.ids) != len(set(self.ids)):
            raise ValueError(f"{csv_path}: duplicate image IDs in {id_column}; verify split")
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
            # Allow CSV ID containing a relative or absolute path; do not silently substitute images.
            if image_id_path.is_absolute():
                candidates.insert(0, image_id_path)
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
    name = re.sub(r"_seed_?\d+$", "", name)
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
    truths, probs = np.concatenate(ys), np.concatenate(ps)
    if not np.isfinite(probs).all():
        raise ValueError("Model output contains NaN or infinite probabilities")
    return truths, probs


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
    """Return aggregate metrics and complete per-label binary classification table.

    Undefined AP/AUROC/F1 for single-class label is reported NaN, never fabricated.
    Macro metrics use available labels and expose the number of valid classes.
    """
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    assert y.shape == p.shape and y.shape[1] == len(labels)
    predicted = (p >= thresholds[None, :]).astype(np.int64)
    records = []
    for c, label in enumerate(labels):
        yi, pi, pr = y[:, c], p[:, c], predicted[:, c]
        n_pos = int(yi.sum())
        n_neg = len(yi) - n_pos
        tp = int(((yi == 1) & (pr == 1)).sum())
        fp = int(((yi == 0) & (pr == 1)).sum())
        tn = int(((yi == 0) & (pr == 0)).sum())
        fn = int(((yi == 1) & (pr == 0)).sum())
        valid = 0 < n_pos < len(yi)
        records.append(dict(
            label=label, positives=n_pos, negatives=n_neg,
            TP=tp, FP=fp, TN=tn, FN=fn,
            AP=float(average_precision_score(yi, pi)) if valid else np.nan,
            AUROC=float(roc_auc_score(yi, pi)) if valid else np.nan,
            Precision=tp/(tp+fp) if (tp+fp) else np.nan,
            Recall=tp/n_pos if n_pos else np.nan,
            Specificity=tn/n_neg if n_neg else np.nan,
            F1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else np.nan,
            Balanced_Accuracy=(tp/n_pos + tn/n_neg)/2 if valid else np.nan,
            threshold=float(thresholds[c]),
        ))
    df = pd.DataFrame(records)
    # Micro-F1 is computed globally across all labels; unlike macro averaging,
    # this remains meaningful when an individual label has no positive examples.
    summary = dict(
        test_images=len(y),
        mAP=float(df.AP.mean()),
        macro_AUROC=float(df.AUROC.mean()),
        macro_F1=float(df.F1.mean()),
        micro_F1=float(f1_score(y.ravel(), predicted.ravel(), zero_division=0)),
        macro_Precision=float(df.Precision.mean()),
        macro_Recall=float(df.Recall.mean()),
        macro_Balanced_Accuracy=float(df.Balanced_Accuracy.mean()),
        valid_AP_labels=int(df.AP.notna().sum()),
        valid_AUROC_labels=int(df.AUROC.notna().sum()),
    )
    return summary, df


def atomic_csv(df: pd.DataFrame, dest: Path):
    """Write CSV atomically to avoid half-written results after interruption."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + '.tmp')
    df.to_csv(temp, index=False, encoding='utf-8-sig')
    temp.replace(dest)


def metric_summary(frame, group_cols, metrics):
    if frame.empty:
        return pd.DataFrame()
    out = frame.groupby(group_cols, dropna=False).agg(
        **{f'{m}_{stat}': (m, stat) for m in metrics for stat in ('mean', 'std')},
        num_seeds=('seed', 'nunique'),
    ).reset_index()
    return out.sort_values(group_cols)


def make_plots(class_df, main_df, out_dir, labels_by_dataset, prefix=''):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not main_df.empty:
        for dataset, df in main_df.groupby('dataset'):
            fig, ax = plt.subplots(figsize=(12, 5))
            means = df.groupby('model')['mAP'].agg(['mean', 'std']).sort_values('mean', ascending=False)
            ax.bar(np.arange(len(means)), means['mean'].values, yerr=means['std'].fillna(0).values, capsize=3)
            ax.set_xticks(np.arange(len(means)), means.index, rotation=50, ha='right')
            ax.set_ylabel('Test mAP')
            ax.set_title(f'{dataset} — Stage 1 Test mAP (mean ± SD across seeds)')
            fig.tight_layout()
            fig.savefig(out_dir / f'{prefix}{dataset}_stage1_test_mAP.png', dpi=180)
            plt.close(fig)
    if not class_df.empty:
        for dataset, df in class_df.groupby('dataset'):
            pivot = df.pivot_table(index='label', columns='model', values='AP', aggfunc='mean')
            pivot = pivot.reindex(labels_by_dataset[dataset])
            fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)*0.75), 7))
            im = ax.imshow(pivot.to_numpy(dtype=float), aspect='auto', vmin=0, vmax=1)
            ax.set_yticks(range(len(pivot)), pivot.index)
            ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=55, ha='right')
            ax.set_title(f'{dataset} — Per-label Test AP, mean across seeds')
            fig.colorbar(im, ax=ax, label='AP')
            fig.tight_layout()
            fig.savefig(out_dir / f'{prefix}{dataset}_stage1_per_class_AP_heatmap.png', dpi=180)
            plt.close(fig)


def main():
    args = parse_args()
    labels_by_dataset = load_labels(args)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    device = torch.device('cuda' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    if not re.fullmatch(r"[a-zA-Z0-9_-]*", args.output_prefix):
        raise ValueError("--output-prefix may only contain letters, numbers, _ or -")
    csv_dir = args.output_dir / 'csv'
    figure_dir = args.output_dir / 'visualizations' 
    csv_dir.mkdir(parents=True, exist_ok=True)
    print(f'[CONFIG] flow={args.flow}, device={device}, checkpoints={args.checkpoints_root}, output={csv_dir}')
    all_rows, class_rows, threshold_rows, errors = [], [], [], []
    plan = []
    coverage = []
    for dataset in args.datasets:
        for seed in args.seeds:
            checkpoints = find_checkpoints(args.checkpoints_root, dataset, seed, args.checkpoint_pattern)
            coverage.append(dict(dataset=dataset, seed=seed, discovered_models=len(checkpoints),
                                 expected_models=args.expected_models, complete=len(checkpoints)==args.expected_models))
            for name, file in checkpoints:
                plan.append((dataset, seed, name, file))
    cov_df = pd.DataFrame(coverage)
    if args.strict_coverage and not cov_df.complete.all():
        raise RuntimeError('Incomplete checkpoint coverage:\n' + cov_df.to_string(index=False))
    if not plan:
        raise SystemExit('No checkpoint found. Check --checkpoints-root and --checkpoint-pattern')
    print(f'[PLAN] {len(plan)} unique checkpoint evaluations')
    if args.strict_coverage and args.checkpoint_pattern != '*' and args.expected_models == 14:
        print('[HINT] Strict coverage expects 14 configurations; for a filtered checkpoint ' 
              'pass --expected-models 1 (or appropriate subset count).')
    # Cache dataset loaders for a single dataset, not all images/predictions.
    for dataset in args.datasets:
        runs = [item for item in plan if item[0] == dataset]
        if not runs:
            continue
        labels = labels_by_dataset[dataset]
        data_dir = args.data_root / dataset
        options = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                       pin_memory=device.type=='cuda')
        try:
            val_loader = DataLoader(XrayCSV(data_dir, 'val', labels, args.image_size), **options)
            test_loader = DataLoader(XrayCSV(data_dir, 'test', labels, args.image_size), **options)
        except Exception as exc:
            if args.fail_fast:
                raise
            errors += [dict(dataset=dataset, seed=seed, model=name, checkpoint=str(path), error=f'dataset: {exc}')
                       for _, seed, name, path in runs]
            continue
        for _, seed, name, path in runs:
            print(f'[RUN] {dataset} | seed{seed} | {name} | {path.name}', flush=True)
            model = None
            try:
                model, payload = build_model(path, device)
                thresholds, source = get_saved_thresholds(payload, path, labels)
                if thresholds is None:
                    y_val, p_val = predict(model, val_loader, device, 'validation')
                    thresholds = validation_thresholds(y_val, p_val)
                    source = 'computed_from_validation'
                y_test, p_test = predict(model, test_loader, device, 'test')
                summary, detail = evaluate_metrics(y_test, p_test, thresholds, labels)
                common = dict(dataset=dataset, model=name, seed=seed, checkpoint=str(path), threshold_source=source)
                all_rows.append({**common, **summary})
                if args.flow in ('detail', 'both'):
                    detail.insert(0, 'seed', seed)
                    detail.insert(0, 'model', name)
                    detail.insert(0, 'dataset', dataset)
                    class_rows.append(detail)
                    threshold_rows.extend(dict(dataset=dataset, model=name, seed=seed, label=l,
                                               threshold=float(t), source=source)
                                          for l, t in zip(labels, thresholds))
                print(f'      mAP={summary["mAP"]:.4f}, macro_F1={summary["macro_F1"]:.4f}', flush=True)
            except Exception as exc:
                errors.append(dict(dataset=dataset, model=name, seed=seed, checkpoint=str(path), error=str(exc)))
                print(f'[ERROR] {dataset}/{name}/seed{seed}: {exc}', file=sys.stderr, flush=True)
                if args.fail_fast:
                    raise
            finally:
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    metrics = ['mAP', 'macro_AUROC', 'macro_F1', 'micro_F1', 'macro_Precision', 'macro_Recall', 'macro_Balanced_Accuracy']
    if all_rows:
        main_df = pd.DataFrame(all_rows).sort_values(['dataset', 'model', 'seed'])
        # Main flow: one row per checkpoint plus mean ± SD per model/dataset.
        if args.flow in ('main', 'both'):
            atomic_csv(main_df, csv_dir / f'{args.output_prefix}all_runs.csv')
            summary_df = metric_summary(main_df, ['dataset', 'model'], metrics)
            atomic_csv(summary_df, csv_dir / f'{args.output_prefix}summary_mean_std.csv')
            print(f'[MAIN] saved prefixed all_runs.csv and summary_mean_std.csv ({len(main_df)} runs)')
        else:
            summary_df = pd.DataFrame()
        if args.flow in ('detail', 'both'):
            if class_rows:
                detail_df = pd.concat(class_rows, ignore_index=True)
                atomic_csv(detail_df, csv_dir / f'{args.output_prefix}per_class.csv')
                atomic_csv(pd.DataFrame(threshold_rows), csv_dir / f'{args.output_prefix}thresholds_validation.csv')
                by_class = detail_df.groupby(['dataset','model','label'], as_index=False).agg(
                    AP_mean=('AP','mean'), AP_std=('AP','std'),
                    AUROC_mean=('AUROC','mean'), F1_mean=('F1','mean'),
                    Recall_mean=('Recall','mean'), num_seeds=('seed','nunique'))
                atomic_csv(by_class, csv_dir / f'{args.output_prefix}per_class_mean_std.csv')
                print(f'[DETAIL] saved prefixed per_class.csv, per_class_mean_std.csv, thresholds_validation.csv')
                make_plots(detail_df, main_df, figure_dir, labels_by_dataset, args.output_prefix)
                print(f'[DETAIL] PNG saved to {figure_dir}')
            else:
                print('[DETAIL] No successful per-class evaluations')
    if errors:
        atomic_csv(pd.DataFrame(errors), csv_dir / f'{args.output_prefix}errors.csv')
        print(f'[WARN] {len(errors)} failed run(s); see {args.output_prefix}errors.csv')
    if not all_rows or errors:
        raise SystemExit(1)
    print(f'[DONE] evaluated {len(all_rows)} checkpoints')


if __name__ == '__main__':
    main()
