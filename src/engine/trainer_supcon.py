from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

from src.engine.metrics import find_optimal_thresholds


def _amp(device: torch.device, enabled: bool):
    return torch.autocast(device_type='cuda', dtype=torch.float16) if enabled and device.type == 'cuda' else nullcontext()


def _require_finite(t: torch.Tensor, name: str):
    if not bool(torch.isfinite(t.detach()).all().item()):
        raise FloatingPointError(f'Nonfinite tensor: {name}')


@dataclass
class TrainResult:
    loss: float
    classification_loss: float
    contrastive_loss: float
    optimizer_steps: int
    skipped_steps: int
    valid_queue_at_end: int
    crossimage_disease_anchors: int
    disease_anchors: int
    seen_images: int


def train_supcon_epoch(model, loader, classifier_loss, contrastive_loss, optimizer, scaler,
                       device, beta: float, accumulation_steps: int = 1,
                       max_batches: Optional[int] = None, amp: bool = True,
                       grad_clip: float = 1.0, epoch: int | None = None,
                       total_epochs: int | None = None, progress_every: int = 10) -> TrainResult:
    if accumulation_steps < 1 or beta < 0:
        raise ValueError('accumulation_steps >= 1 and beta >= 0 required')
    if max_batches is not None and max_batches < 1:
        raise ValueError('max_batches must be >= 1')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    count_batches = len(loader)
    if count_batches == 0:
        raise ValueError('Empty train loader')
    if max_batches is not None:
        count_batches = min(count_batches, max_batches)
    sums = {'loss': 0.0, 'cls': 0.0, 'con': 0.0, 'images': 0,
            'disease': 0, 'cross': 0, 'steps': 0, 'skipped': 0}
    desc = f"TRAIN {epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else "TRAIN"
    pbar = tqdm(loader, total=count_batches, desc=desc, unit="batch", dynamic_ncols=True,
                mininterval=1.0, leave=True)
    for i, (view_q, view_k, labels) in enumerate(pbar):
        if i >= count_batches:
            break
        view_q = view_q.to(device, non_blocking=True)
        view_k = view_k.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        # Correct normalization even if final accumulation group is shorter.
        group_start = (i // accumulation_steps) * accumulation_steps
        group_len = min(accumulation_steps, count_batches - group_start)
        with _amp(device, amp):
            logits, q, k = model(view_q, view_k)
            cls = classifier_loss(logits, labels)
            if beta != 0:
                queue_f, queue_l = model.get_valid_queue()
                con, stats = contrastive_loss(q, k, labels, queue_f, queue_l, return_stats=True)
            else:
                con = q.float().sum() * 0.0
                stats = {'n_disease_anchors': int((labels.sum(1) > 0).sum().item()),
                         'n_disease_with_crossimage_positive': 0}
            total = cls.float() + beta * con.float()
        _require_finite(total, 'total train loss')
        scaler.scale(total / group_len).backward()
        # Enqueue only after evaluating this batch, to avoid self-leakage in queue.
        # Queue stores momentum keys, and will be used by subsequent microbatches.
        with torch.no_grad():
            model.dequeue_and_enqueue(k, labels)
        current_batch = labels.shape[0]
        sums['images'] += current_batch
        sums['loss'] += float(total.detach()) * current_batch
        sums['cls'] += float(cls.detach()) * current_batch
        sums['con'] += float(con.detach()) * current_batch
        sums['disease'] += stats['n_disease_anchors']
        sums['cross'] += stats['n_disease_with_crossimage_positive']

        if (i + 1) % accumulation_steps == 0 or i + 1 == count_batches:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not bool(torch.isfinite(norm).item()):
                raise FloatingPointError('Nonfinite gradient norm')
            before_scale = scaler.get_scale() if scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            did_step = (not scaler.is_enabled()) or scaler.get_scale() >= before_scale
            if did_step:
                model.update_momentum_encoder()
                sums['steps'] += 1
            else:
                sums['skipped'] += 1
            optimizer.zero_grad(set_to_none=True)
        if (i + 1) % max(1, progress_every) == 0 or i + 1 == count_batches:
            mem = f"{torch.cuda.memory_allocated(device) / (1024**3):.2f}G" if device.type == 'cuda' else 'CPU'
            pbar.set_postfix({
                'loss': f"{sums['loss']/sums['images']:.4f}",
                'bce': f"{sums['cls']/sums['images']:.4f}",
                'supcon': f"{sums['con']/sums['images']:.4f}",
                'queue': int(model.queue_count.item()),
                'vram': mem,
            }, refresh=False)
    pbar.close()
    if sums['images'] == 0:
        raise RuntimeError('No images trained')
    return TrainResult(
        loss=sums['loss']/sums['images'],
        classification_loss=sums['cls']/sums['images'],
        contrastive_loss=sums['con']/sums['images'],
        optimizer_steps=sums['steps'], skipped_steps=sums['skipped'],
        valid_queue_at_end=int(model.queue_count.item()),
        crossimage_disease_anchors=sums['cross'], disease_anchors=sums['disease'],
        seen_images=sums['images'],
    )


@torch.no_grad()
def validate_supcon(model, loader, classifier_loss, device, amp=True, max_batches=None,
                    epoch: int | None = None, total_epochs: int | None = None,
                    progress_every: int = 10):
    """Val only: AP and AUC omit single-class labels; thresholds fitted to val only.

    Macro F1, precision, recall computed on same valid-class mask for stable
    comparability. Invalid labels return NaN per label, not artificially zero AP.
    """
    model.eval()
    all_targets, all_scores = [], []
    loss_sum, n = 0.0, 0
    val_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    desc = f"VAL {epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else "VAL"
    pbar = tqdm(loader, total=val_batches, desc=desc, unit="batch", dynamic_ncols=True,
                mininterval=1.0, leave=True)
    for i, (images, labels) in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        with _amp(device, amp):
            logits = model(images)
            loss = classifier_loss(logits, labels)
            probs = torch.sigmoid(logits.float())
        _require_finite(probs, 'validation probabilities')
        bs = labels.shape[0]
        loss_sum += float(loss.detach())*bs
        n += bs
        all_targets.append(labels.cpu().numpy())
        all_scores.append(probs.cpu().numpy())
        if (i + 1) % max(1, progress_every) == 0 or i + 1 == val_batches:
            pbar.set_postfix({'val_loss': f'{loss_sum/n:.4f}', 'images': n}, refresh=False)
    pbar.close()
    if not n:
        raise ValueError('Validation loader yielded no samples')
    y_true = np.concatenate(all_targets, axis=0).astype(np.int64)
    y_prob = np.concatenate(all_scores, axis=0).astype(np.float64)
    valid = np.array([len(np.unique(y_true[:, j])) == 2 for j in range(y_true.shape[1])])
    thresholds = find_optimal_thresholds(y_true, y_prob).astype(float)
    if valid.any():
        aps = np.array([average_precision_score(y_true[:, j],y_prob[:, j]) if valid[j] else np.nan
                        for j in range(y_true.shape[1])], dtype=float)
        aucs = np.array([roc_auc_score(y_true[:, j],y_prob[:, j]) if valid[j] else np.nan
                         for j in range(y_true.shape[1])], dtype=float)
        yt, yp = y_true[:,valid], (y_prob[:,valid]>=thresholds[valid]).astype(np.int64)
        metrics = dict(mAP=float(np.nanmean(aps)), macro_AUC=float(np.nanmean(aucs)),
                       macro_F1=float(f1_score(yt,yp,average='macro',zero_division=0)),
                       macro_Precision=float(precision_score(yt,yp,average='macro',zero_division=0)),
                       macro_Recall=float(recall_score(yt,yp,average='macro',zero_division=0)))
    else:
        aps = np.full(y_true.shape[1], np.nan)
        metrics = dict(mAP=float('nan'), macro_AUC=float('nan'), macro_F1=float('nan'),
                       macro_Precision=float('nan'), macro_Recall=float('nan'))
    return dict(val_loss=loss_sum/n, n_images=n, n_valid_classes=int(valid.sum()),
                thresholds=thresholds.tolist(), per_class_AP=aps.tolist(), **metrics)
