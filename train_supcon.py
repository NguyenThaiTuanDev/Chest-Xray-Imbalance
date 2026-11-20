from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

SEEDS = (7, 42, 132, 2024, 3407)
DATASETS = ("VinDr-CXR", "VinDr-PCXR")

PRESETS = {
    "bce":       dict(loss="bce"),
    "wbce_w0":   dict(loss="wbce", wbce_mode=0),
    "wbce_w1":   dict(loss="wbce", wbce_mode=1),
    "wbce_w2":   dict(loss="wbce", wbce_mode=2),
    "wbce_w3":   dict(loss="wbce", wbce_mode=3),
    "focal_f0":  dict(loss="focal", gamma=0.0, alpha=None),
    "focal_f1":  dict(loss="focal", gamma=2.0, alpha=0.25),
    "focal_f2":  dict(loss="focal", gamma=2.0, alpha=None),
    "focal_f3":  dict(loss="focal", gamma=1.0, alpha=None),
    "focal_f4":  dict(loss="focal", gamma=2.0, alpha=0.75),
    "asl_a0":    dict(loss="asl", gamma_pos=0.0, gamma_neg=0.0, margin=0.0),
    "asl_a1":    dict(loss="asl", gamma_pos=0.0, gamma_neg=4.0, margin=0.05),
    "asl_a2":    dict(loss="asl", gamma_pos=0.0, gamma_neg=2.0, margin=0.05),
    "asl_a3":    dict(loss="asl", gamma_pos=0.0, gamma_neg=4.0, margin=0.0),
}
METRIC_NAMES = ("mAP", "macro_AUC", "macro_F1", "macro_Precision", "macro_Recall")


def log(message):
    print(message, flush=True)


def atomic_torch_save(payload, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    if tmp.exists():
        tmp.unlink()  # A failed incomplete write, not an existing checkpoint.
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(tmp, path)


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def read_history(path):
    if not Path(path).is_file():
        return []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_loss(args, train_labels, device):
    import numpy as np
    import torch
    from src.losses.losses import BCELoss, WBCELoss, FocalLossMultiLabel, AsymmetricLoss
    cfg = PRESETS[args.preset]
    if cfg["loss"] == "bce":
        return BCELoss().to(device)
    if cfg["loss"] == "wbce":
        mode = cfg["wbce_mode"]
        counts = train_labels.sum(axis=0).astype(float)
        n = len(train_labels)
        ratio = (n - counts) / np.maximum(counts, 1.0)
        weights = {0: lambda: np.ones_like(ratio),
                   1: lambda: np.sqrt(ratio),
                   2: lambda: np.clip(ratio, 0, 10),
                   3: lambda: np.clip(ratio, 0, 50)}[mode]()
        log(f"[WBCE] mode={mode}; pos_weight={weights.round(4).tolist()}")
        return WBCELoss(torch.tensor(weights, dtype=torch.float32)).to(device)
    if cfg["loss"] == "focal":
        return FocalLossMultiLabel(gamma=cfg["gamma"], alpha=cfg["alpha"]).to(device)
    return AsymmetricLoss(gamma_pos=cfg["gamma_pos"],
                          gamma_neg=cfg["gamma_neg"], margin=cfg["margin"]).to(device)


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested):
    import torch
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        log("[WARN] CUDA unavailable; falling back to CPU (very slow for DenseNet201)")
        return torch.device("cpu")
    log(f"[GPU] {torch.cuda.get_device_name(0)} | GPUs detected={torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        log("[NOTE] This local trainer uses CUDA:0 only; no DataParallel to protect the momentum queue.")
    return torch.device("cuda:0")


def validate_data(dataset_dir):
    paths = [dataset_dir / "labels" / "train_split.csv",
             dataset_dir / "labels" / "val_split.csv", dataset_dir / "train"]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset inputs:\n" + "\n".join(missing))


def config_signature(args):
    return dict(dataset=args.dataset, preset=args.preset, seed=args.seed,
                loss_parameters=PRESETS[args.preset], epochs=args.epochs,
                batch_size=args.batch_size, accumulation_steps=args.accumulation_steps,
                image_size=args.image_size, queue_size=args.queue_size,
                beta=args.beta, momentum=args.momentum, temperature=args.temperature,
                proj_dim=args.proj_dim, lr=args.lr, weight_decay=args.weight_decay,
                amp=not args.no_amp, pretrained=not args.no_pretrained)


def run_single(args):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import VinDrDataset
    from src.data.transforms import get_transforms
    from src.data.contrastive_dataset import ContrastiveVinDrDataset
    from src.models.builder import build_model
    from src.losses.contrastive_loss import MultiLabelQueueContrastiveLoss
    from src.engine.trainer_supcon import train_supcon_epoch, validate_supcon

    set_seed(args.seed)
    device = choose_device(args.device)
    root = args.output_root.resolve()
    data_dir = args.data_root.resolve() / args.dataset
    validate_data(data_dir)
    final = root / "checkpoints" / args.dataset / f"seed{args.seed}" / f"best_{args.preset}_supcon_seed{args.seed}.pth"
    work = root / "_work" / args.dataset / args.preset / f"seed{args.seed}"
    latest = work / "latest.pth"
    best = work / "best.pth"
    history = work / "history.csv"
    config_path = work / "config.json"
    report_path = root / "csv" / "training_history.csv"
    signature = config_signature(args)
    if final.exists() and not args.overwrite:
        log(f"[SKIP] Existing BEST, preserved: {final}")
        return
    if final.exists() and args.overwrite and not args.confirm_overwrite:
        raise ValueError("Overwriting existing BEST requires BOTH --overwrite and --confirm-overwrite")
    if args.resume and not latest.is_file():
        raise FileNotFoundError(f"--resume requested but no local latest checkpoint: {latest}")
    if latest.exists() and not args.resume:
        raise FileExistsError(f"Partial run exists: {latest}; use --resume (recommended) or --discard-partial")
    if work.exists() and not latest.exists() and not args.discard_partial and not args.resume:
        leftovers = [x.name for x in work.iterdir() if x.name != "config.json"]
        if leftovers:
            raise FileExistsError(f"Incomplete work folder: {work}; files={leftovers}. Inspect before using --discard-partial")
        # A config-only folder is safe to retry from epoch 1 after an early crash.
    if args.discard_partial:
        if args.resume:
            raise ValueError("--resume and --discard-partial conflict")
        if work.exists():
            shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != signature:
            raise ValueError("Saved config differs from current command; refusing incompatible resume")
    else:
        atomic_json_save(signature, config_path)

    log(f"[DATA] Loading {data_dir}")
    train_ds = ContrastiveVinDrDataset(data_dir, args.dataset, args.image_size, strict_images=True)
    val_ds = VinDrDataset(csv_file=data_dir / "labels" / "val_split.csv",
                          img_dir=data_dir / "train", dataset_name=args.dataset,
                          transform=get_transforms("val", args.image_size), strict_images=True)
    generator = torch.Generator()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              generator=generator, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            persistent_workers=False)
    log(f"[DATA READY] train={len(train_ds)} | val={len(val_ds)} | steps/epoch={len(train_loader)}")
    log("[MODEL] Loading ImageNet DenseNet201 + momentum encoder ...")
    model = build_model("lcl_densenet201", num_classes=14,
                        pretrained=not args.no_pretrained,
                        proj_dim=args.proj_dim, queue_size=args.queue_size,
                        momentum=args.momentum).to(device)
    loss_cls = get_loss(args, train_ds.labels.numpy(), device)
    loss_con = MultiLabelQueueContrastiveLoss(temperature=args.temperature).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                   lr=args.lr, weight_decay=args.weight_decay)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    start_epoch, best_map, best_epoch = 1, float("-inf"), 0
    if args.resume:
        log(f"[RESUME] Loading {latest}")
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        if ck["signature"] != signature:
            raise ValueError("Latest checkpoint is incompatible with this run")
        model.load_state_dict(ck["model_state"], strict=True)
        optimizer.load_state_dict(ck["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scaler.load_state_dict(ck["scaler_state"])
        torch.set_rng_state(ck["rng_torch"].cpu())
        np.random.set_state(ck["rng_numpy"])
        random.setstate(ck["rng_python"])
        if device.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        start_epoch = int(ck["epoch"]) + 1
        best_map, best_epoch = float(ck["best_val_map"]), int(ck["best_epoch"])
        stored = read_history(history)
        if len(stored) == start_epoch - 2 and ck.get("epoch_row"):
            append_csv(history, ck["epoch_row"])
        elif len(stored) != start_epoch - 1:
            raise RuntimeError("History and latest checkpoint disagree: manual recovery required")
        if best_epoch:
            if not best.is_file():
                raise FileNotFoundError(f"Best partial checkpoint missing: {best}")
            best_meta = torch.load(best, map_location="cpu", weights_only=False)
            if int(best_meta.get("epoch", -1)) != best_epoch:
                raise RuntimeError("Best partial and latest checkpoint disagree (possible crash between saves). "
                                   "Do not resume automatically; inspect the run to prevent incorrect BEST.")
            del best_meta
        log(f"[RESUMED] starting epoch {start_epoch}; best_val_mAP={best_map:.5f}")
        del ck

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.monotonic()
        generator.manual_seed(args.seed + epoch)
        log(f"\n{'='*70}\n[TRAIN] {args.dataset} {args.preset} seed{args.seed} | epoch {epoch}/{args.epochs}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = train_supcon_epoch(model, train_loader, loss_cls, loss_con, optimizer, scaler,
                                   device, beta=args.beta,
                                   accumulation_steps=args.accumulation_steps,
                                   amp=(device.type == "cuda" and not args.no_amp),
                                   progress_every=args.progress_every,
                                   epoch=epoch, total_epochs=args.epochs)
        log(f"[VAL] {args.dataset} {args.preset} seed{args.seed} | epoch {epoch}/{args.epochs}")
        val = validate_supcon(model, val_loader, loss_cls, device,
                              amp=(device.type == "cuda" and not args.no_amp),
                              epoch=epoch, total_epochs=args.epochs,
                              progress_every=args.progress_every)
        val_map = float(val["mAP"])
        improved = bool(np.isfinite(val_map) and val_map > best_map)
        if improved:
            best_map, best_epoch = val_map, epoch
            atomic_torch_save(dict(model_state=model.state_dict(),
                                   epoch=epoch, best_val_map=best_map,
                                   validation=val, config=signature,
                                   dataset=args.dataset, seed=args.seed,
                                   loss_tag=args.preset), best)
        vram = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3) if device.type == "cuda" else 0.
        row = dict(dataset=args.dataset, loss_tag=args.preset, seed=args.seed,
                   epoch=epoch, train_loss=result.loss,
                   cls_loss=result.classification_loss, supcon_loss=result.contrastive_loss,
                   val_loss=val["val_loss"],
                   **{k:val[k] for k in METRIC_NAMES},
                   best_val_mAP=best_map, best_epoch=best_epoch,
                   queue_count=result.valid_queue_at_end, optimizer_steps=result.optimizer_steps,
                   skipped_steps=result.skipped_steps, train_images_seen=result.seen_images,
                   vram_peak_gib=vram, epoch_seconds=round(time.monotonic()-t0, 1))
        payload = dict(epoch=epoch, model_state=model.state_dict(),
                       optimizer_state=optimizer.state_dict(), scaler_state=scaler.state_dict(),
                       signature=signature, best_val_map=best_map, best_epoch=best_epoch,
                       rng_torch=torch.get_rng_state(), rng_numpy=np.random.get_state(),
                       rng_python=random.getstate(),
                       rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                       epoch_row=row)
        atomic_torch_save(payload, latest)
        append_csv(history, row)
        log(f"[EPOCH DONE] {epoch}/{args.epochs} | Val mAP={val_map:.5f} | "
            f"best={best_map:.5f} (epoch {best_epoch}) | VRAM_peak={vram:.2f} GiB | "
            f"{row['epoch_seconds']}s")

    if not best.is_file():
        raise RuntimeError("No finite Validation mAP; refusing to export a BEST checkpoint")
    ck = torch.load(best, map_location="cpu", weights_only=False)
    if ck.get("config") != signature or ck.get("epoch") != best_epoch:
        raise RuntimeError("Final best checkpoint metadata mismatch")
    tmp_final = final.with_suffix(".pth.writing")
    final.parent.mkdir(parents=True, exist_ok=True)
    if tmp_final.exists():
        tmp_final.unlink()
    shutil.copyfile(best, tmp_final)
    if tmp_final.stat().st_size != best.stat().st_size:
        tmp_final.unlink()
        raise IOError("Incomplete BEST checkpoint copy")
    os.replace(tmp_final, final)
    log(f"[BEST] {final} | best epoch {best_epoch}/{args.epochs}, Val mAP={best_map:.5f}")
    # Save per-epoch data as one independent flat CSV (no nested CSV directories).
    rows = read_history(history)
    for entry in rows:
        append_csv(report_path, entry)
    atomic_json_save(dict(dataset=args.dataset, seed=args.seed, loss_tag=args.preset,
                          best_epoch=best_epoch, best_val_mAP=best_map,
                          checkpoint=str(final), complete=True),
                     root / "csv" / f"{args.dataset}_{args.preset}_seed{args.seed}_summary.json")
    shutil.rmtree(work)
    log("[DONE] Only one published BEST .pth; temporary resume files cleared")


def build_parser():
    p = argparse.ArgumentParser(description="Local Stage 2: full 14 Loss + SupCon x 5 seeds x 2 datasets")
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/stage2"))
    p.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    p.add_argument("--preset", choices=("all", *PRESETS), default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=8)
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0, help="Windows safest: 0")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--queue-size", type=int, default=1024)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--momentum", type=float, default=0.999)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true", help="Debug only, not comparable to ImageNet experiment")
    p.add_argument("--resume", action="store_true", help="Resume partial run from local _work/latest.pth")
    p.add_argument("--auto-resume", action="store_true", help="Outer loop resumes partial runs automatically")
    p.add_argument("--discard-partial", action="store_true", help="Explicitly discard only this run's local temporary state")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--confirm-overwrite", action="store_true")
    p.add_argument("--list", action="store_true", help="Print plan and presets only")
    p.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    return p


def main():
    args = build_parser().parse_args()
    if any(x <= 0 for x in (args.epochs, args.batch_size, args.val_batch_size,
                            args.accumulation_steps, args.image_size, args.queue_size,
                            args.progress_every)):
        raise ValueError("Epoch, batch, accumulation, size, queue, progress must be positive")
    if not 0 < args.temperature <= 1 or args.beta < 0 or not 0 <= args.momentum < 1:
        raise ValueError("Invalid temperature, beta or momentum")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seed IDs")
    plan = [(d, p, s) for d in (DATASETS if args.dataset == "all" else [args.dataset])
            for p in (PRESETS if args.preset == "all" else [args.preset])
            for s in args.seeds]
    if args.list:
        for i, (d,p,s) in enumerate(plan, 1):
            log(f"[PLAN] {i:03}/{len(plan):03} {d} {p} seed{s} {PRESETS[p]}")
        log(f"[TOTAL] {len(plan)} runs. Default batch={args.batch_size} x accum={args.accumulation_steps} (SupCon sees {args.batch_size} per batch)")
        return 0
    if args.single:
        if len(plan) != 1:
            raise ValueError("Internal --single requires exactly one dataset, preset and seed")
        run_single(args)
        return 0

    script = Path(__file__).resolve()
    log(f"[START] {len(plan)} run(s), output={args.output_root.resolve()}")
    log(f"[CONFIG] batch={args.batch_size} accum={args.accumulation_steps} | "
        f"SupCon per-batch positives based on actual batch={args.batch_size}")
    for index, (dataset, preset, seed) in enumerate(plan, 1):
        output = args.output_root.resolve()
        best = output / "checkpoints" / dataset / f"seed{seed}" / f"best_{preset}_supcon_seed{seed}.pth"
        if best.exists() and not args.overwrite:
            log(f"[SKIP] {index}/{len(plan)} {dataset} {preset} seed{seed}: BEST exists")
            continue
        work_latest = output / "_work" / dataset / preset / f"seed{seed}" / "latest.pth"
        cmd = [sys.executable, "-u", str(script), "--single", "--dataset", dataset,
               "--preset", preset, "--seeds", str(seed)]
        # Forward only known args without duplicating selected dimensions.
        for flag, value in (("--data-root", args.data_root), ("--output-root", args.output_root),
                            ("--epochs", args.epochs), ("--batch-size", args.batch_size),
                            ("--val-batch-size", args.val_batch_size),
                            ("--accumulation-steps", args.accumulation_steps),
                            ("--num-workers", args.num_workers), ("--image-size", args.image_size),
                            ("--queue-size", args.queue_size), ("--proj-dim", args.proj_dim),
                            ("--momentum", args.momentum), ("--temperature", args.temperature),
                            ("--beta", args.beta), ("--lr", args.lr),
                            ("--weight-decay", args.weight_decay),
                            ("--progress-every", args.progress_every), ("--device", args.device)):
            cmd.extend((flag, str(value)))
        for flag in ("no_amp", "no_pretrained", "overwrite", "confirm_overwrite"):
            if getattr(args, flag):
                cmd.append("--" + flag.replace("_", "-"))
        if args.discard_partial:
            cmd.append("--discard-partial")
        elif args.resume or (args.auto_resume and work_latest.is_file()):
            cmd.append("--resume")
        log(f"\n[RUN] {index}/{len(plan)} {dataset} {preset} seed{seed}")
        ret = subprocess.run(cmd, cwd=str(script.parent), check=False)
        if ret.returncode:
            log(f"[FAILED] {dataset} {preset} seed{seed} | exit={ret.returncode}")
            log(f"[RECOVER] After addressing the error, use --auto-resume to resume completed epochs.")
            return ret.returncode
        gc.collect()
    log(f"[DONE] All selected run(s) processed. Check {args.output_root.resolve() / 'checkpoints'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
