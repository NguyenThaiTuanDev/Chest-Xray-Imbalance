"""Multi-label momentum contrastive loss for VinDr-CXR and VinDr-PCXR.

Algorithmic decision for initial experiment:
  - Each query has one compulsory positive: its own momentum key.
  - Different disease images are positives if they share at least one positive label.
  - All-zero images are NOT mutually positive.
  - Other valid keys (including all-zero images) enter the denominator.
  - Only valid queued keys are passed, never unfilled queue slots.

This is a weighted supervised contrastive objective; it is not identical to
single-label SupCon or to original MoCo.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MultiLabelQueueContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, class_weights: Tensor | None = None):
        super().__init__()
        if not 0 < temperature <= 1:
            raise ValueError("temperature must be in (0, 1]")
        self.temperature = float(temperature)
        if class_weights is not None:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.ndim != 1 or weights.numel() == 0 or not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("class_weights must be a positive, finite 1-D vector")
            self.register_buffer("class_weights", weights.clone())
        else:
            self.class_weights = None

    @staticmethod
    def _validate_binary(x: Tensor, name: str) -> None:
        if x.ndim != 2 or not torch.isfinite(x).all() or not torch.all((x == 0) | (x == 1)):
            raise ValueError(f"{name} must be a finite binary 2-D matrix")

    def build_positive_weights(self, query_labels: Tensor, dictionary_labels: Tensor, batch_size: int) -> Tensor:
        """Return [B, B+K] weights; matching self-key slots i,i have weight=1.

        Weight for two distinct images is weighted cosine of their positive
        multi-hot vectors, and is zero for unrelated/all-zero images.
        """
        self._validate_binary(query_labels, "query_labels")
        self._validate_binary(dictionary_labels, "dictionary_labels")
        if query_labels.shape[1] != dictionary_labels.shape[1]:
            raise ValueError("Class dimension mismatch")
        if batch_size != query_labels.shape[0] or dictionary_labels.shape[0] < batch_size:
            raise ValueError("Dictionary must start with current batch keys")
        q, d = query_labels.float(), dictionary_labels.float()
        if self.class_weights is not None:
            if self.class_weights.numel() != q.shape[1]:
                raise ValueError("class_weights length != number of classes")
            w = self.class_weights.to(q.device)
            q = q * w.sqrt()
            d = d * w.sqrt()
        qn = F.normalize(q, p=2, dim=1, eps=1e-8)
        dn = F.normalize(d, p=2, dim=1, eps=1e-8)
        sim = (qn @ dn.T).clamp(0, 1)
        ix = torch.arange(batch_size, device=sim.device)
        sim[ix, ix] = 1.0
        return sim

    def forward(self, q: Tensor, k: Tensor, labels: Tensor,
                queue_features: Tensor | None = None,
                queue_labels: Tensor | None = None,
                return_stats: bool = False):
        """q/k: [B,D], labels: [B,C]; queue features: [D,K], labels: [C,K].

        If K=0 pass None or correctly shaped zero-column tensors.
        Loss accumulates in fp32 even under autocast. Key and queue are detached.
        """
        if q.ndim != 2 or k.ndim != 2 or q.shape != k.shape or q.shape[0] == 0:
            raise ValueError("q and k must have identical nonempty [B,D] shapes")
        self._validate_binary(labels, "labels")
        if labels.shape[0] != q.shape[0] or labels.device != q.device or k.device != q.device:
            raise ValueError("Batch/device mismatch")
        b, d = q.shape
        c = labels.shape[1]
        if (queue_features is None) != (queue_labels is None):
            raise ValueError("queue_features and queue_labels must both be set or both None")
        if queue_features is None:
            queue_features = q.new_empty((d, 0))
            queue_labels = labels.new_empty((c, 0))
        if queue_features.shape[0] != d or queue_labels.shape[0] != c or queue_features.ndim != 2 or queue_labels.ndim != 2 or queue_features.shape[1] != queue_labels.shape[1]:
            raise ValueError("Queue must be [D,K] and [C,K]")
        if queue_features.device != q.device or queue_labels.device != q.device:
            raise ValueError("Queue/device mismatch")
        if not torch.isfinite(q).all() or not torch.isfinite(k).all() or not torch.isfinite(queue_features).all():
            raise ValueError("Nonfinite embeddings")
        self._validate_binary(queue_labels, "queue_labels")

        q32 = F.normalize(q.float(), dim=1)
        k32 = F.normalize(k.detach().float(), dim=1)
        bank = torch.cat([k32, F.normalize(queue_features.T.detach().float(), dim=1)], dim=0)
        dict_labels = torch.cat([labels.float(), queue_labels.T.float()], dim=0)
        weights = self.build_positive_weights(labels, dict_labels, b)

        log_prob = F.log_softmax((q32 @ bank.T) / self.temperature, dim=1)
        positive_mass = weights.sum(1)
        loss = -((weights * log_prob).sum(1) / positive_mass.clamp_min(1e-8)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite contrastive loss")
        if not return_stats:
            return loss
        other = weights.clone()
        ix = torch.arange(b, device=q.device)
        other[ix, ix] = 0
        disease = labels.sum(1) > 0
        stats = {
            "batch_size": b,
            "queue_size_valid": int(queue_features.shape[1]),
            "n_disease_anchors": int(disease.sum().item()),
            "n_allzero_anchors": int((~disease).sum().item()),
            "n_crossimage_positive_anchors": int((other.sum(1) > 0).sum().item()),
            "n_disease_with_crossimage_positive": int(((other.sum(1) > 0) & disease).sum().item()),
            "mean_positive_mass": float(positive_mass.mean().detach().cpu()),
        }
        return loss, stats
