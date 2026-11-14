
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):

        return self.loss_fn(
            logits.float(),
            targets.float()
        )


class WBCELoss(nn.Module):

    def __init__(self, pos_weights):

        super().__init__()

        self.register_buffer(
            "pos_weights",
            torch.as_tensor(
                pos_weights,
                dtype=torch.float32
            )
        )

    def forward(self, logits, targets):

        return F.binary_cross_entropy_with_logits(
            logits.float(),
            targets.float(),
            pos_weight=self.pos_weights
        )


class FocalLossMultiLabel(nn.Module):

    def __init__(self, gamma=2.0, alpha=0.25):

        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):

        logits = logits.float()
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        pt = torch.exp(-bce_loss)

        focal_loss = (
            (1.0 - pt).pow(self.gamma)
            * bce_loss
        )

        if self.alpha is not None:

            alpha_t = (
                self.alpha * targets
                + (1.0 - self.alpha) * (1.0 - targets)
            )

            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class AsymmetricLoss(nn.Module):

    def __init__(
        self,
        gamma_pos=0.0,
        gamma_neg=4.0,
        margin=0.05,
        eps=1e-8
    ):

        super().__init__()

        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.eps = eps

    def forward(self, logits, targets):

        logits = logits.float()
        targets = targets.float()

        if (
            self.gamma_pos == 0.0
            and self.gamma_neg == 0.0
            and self.margin == 0.0
        ):

            return F.binary_cross_entropy_with_logits(
                logits,
                targets
            )

        probs_pos = torch.sigmoid(logits)

        probs_neg = 1.0 - probs_pos

        if self.margin > 0:

            probs_neg = (
                probs_neg + self.margin
            ).clamp(max=1.0)

        probs_pos = probs_pos.clamp(
            min=self.eps,
            max=1.0 - self.eps
        )

        probs_neg = probs_neg.clamp(
            min=self.eps,
            max=1.0 - self.eps
        )

        loss_pos = targets * torch.log(probs_pos)

        loss_neg = (
            (1.0 - targets)
            * torch.log(probs_neg)
        )

        if (
            self.gamma_pos > 0
            or self.gamma_neg > 0
        ):

            with torch.no_grad():

                pt_pos = probs_pos * targets

                pt_neg = probs_neg * (1.0 - targets)

                pt = pt_pos + pt_neg

                gamma = (
                    self.gamma_pos * targets
                    + self.gamma_neg * (1.0 - targets)
                )

                weight = (1.0 - pt).pow(gamma)

            loss_pos = loss_pos * weight
            loss_neg = loss_neg * weight

        loss = -(loss_pos + loss_neg)

        return loss.mean()


class MultiLabelMoCoLoss(nn.Module):
    """
    Multi-Label Label Contrastive Learning Loss.

    Positive pairs:
    1. Query and its corresponding Key.
    2. Two normal images (all-zero labels).
    3. Images with overlapping disease labels.

    Negative pairs:
    1. Normal and disease images.
    2. Disease images without overlapping labels.

    Similarity between disease labels:
    Multi-label Cosine Similarity.
    """

    def __init__(
        self,
        temperature=0.07,
        eps=1e-8
    ):

        super().__init__()

        if temperature <= 0:
            raise ValueError(
                "Temperature must be positive."
            )

        self.temperature = temperature
        self.eps = eps

    def build_label_similarity(
        self,
        query_labels,
        dictionary_labels,
        batch_size
    ):

        query_labels = query_labels.float()

        dictionary_labels = dictionary_labels.float()

        if query_labels.ndim != 2:
            raise ValueError(
                "query_labels must have shape (B, C)."
            )

        if dictionary_labels.ndim != 2:
            raise ValueError(
                "dictionary_labels must have shape (N, C)."
            )

        if query_labels.shape[1] != dictionary_labels.shape[1]:
            raise ValueError(
                "Query and dictionary class dimensions mismatch."
            )

        if batch_size != query_labels.shape[0]:
            raise ValueError(
                "Invalid batch_size."
            )

        query_is_normal = (
            query_labels.sum(dim=1) == 0
        )

        dictionary_is_normal = (
            dictionary_labels.sum(dim=1) == 0
        )

        query_norm = F.normalize(
            query_labels,
            p=2,
            dim=1,
            eps=self.eps
        )

        dictionary_norm = F.normalize(
            dictionary_labels,
            p=2,
            dim=1,
            eps=self.eps
        )

        similarity = torch.matmul(
            query_norm,
            dictionary_norm.T
        )

        # Normal-to-Normal Positive
        normal_pairs = (
            query_is_normal.unsqueeze(1)
            & dictionary_is_normal.unsqueeze(0)
        )

        similarity = torch.where(
            normal_pairs,
            torch.ones_like(similarity),
            similarity
        )

        # Normal-to-Disease Negative
        normal_disease_pairs = (
            query_is_normal.unsqueeze(1)
            | dictionary_is_normal.unsqueeze(0)
        )

        similarity = torch.where(
            normal_disease_pairs & ~normal_pairs,
            torch.zeros_like(similarity),
            similarity
        )

        # Same-image Query-Key Positive
        # The first B dictionary entries are current batch keys.
        indices = torch.arange(
            batch_size,
            device=query_labels.device
        )

        similarity[
            indices,
            indices
        ] = 1.0

        return similarity.clamp(
            min=0.0,
            max=1.0
        )

    def forward(
        self,
        q,
        k,
        labels,
        queue_features,
        queue_labels
    ):

        q = q.float()
        k = k.float()
        labels = labels.float()

        queue_features = queue_features.float()
        queue_labels = queue_labels.float()

        if q.ndim != 2 or k.ndim != 2:
            raise ValueError(
                "q and k must have shape (B, D)."
            )

        if q.shape != k.shape:
            raise ValueError(
                "Query and Key shapes must match."
            )

        batch_size, feature_dim = q.shape

        if labels.shape[0] != batch_size:
            raise ValueError(
                "Labels and embeddings batch sizes mismatch."
            )

        if queue_features.ndim != 2:
            raise ValueError(
                "queue_features must have shape (D, K)."
            )

        if queue_labels.ndim != 2:
            raise ValueError(
                "queue_labels must have shape (C, K)."
            )

        if queue_features.shape[0] != feature_dim:
            raise ValueError(
                "Queue feature dimension mismatch."
            )

        if queue_features.shape[1] != queue_labels.shape[1]:
            raise ValueError(
                "Queue feature and label counts mismatch."
            )

        if queue_labels.shape[0] != labels.shape[1]:
            raise ValueError(
                "Queue label dimension mismatch."
            )

        # Normalize embeddings
        q = F.normalize(
            q,
            p=2,
            dim=1
        )

        k = F.normalize(
            k,
            p=2,
            dim=1
        )

        # Current keys + Memory Bank
        dictionary_features = torch.cat(
            [
                k,
                queue_features.T
            ],
            dim=0
        )

        dictionary_labels = torch.cat(
            [
                labels,
                queue_labels.T
            ],
            dim=0
        )

        # Multi-label similarity matrix
        label_similarity = self.build_label_similarity(
            query_labels=labels,
            dictionary_labels=dictionary_labels,
            batch_size=batch_size
        )

        # Feature similarity
        logits = torch.matmul(
            q,
            dictionary_features.T
        ) / self.temperature

        # Numerical stability
        log_prob = F.log_softmax(
            logits,
            dim=1
        )

        # Weighted positive log probability
        positive_weights = label_similarity

        positive_sum = positive_weights.sum(
            dim=1,
            keepdim=True
        )

        # Every query has at least its own Key as positive
        if torch.any(positive_sum <= self.eps):
            raise RuntimeError(
                "A query has no valid positive pair."
            )

        mean_positive_log_prob = (
            positive_weights * log_prob
        ).sum(dim=1) / positive_sum.squeeze(1)

        loss = -mean_positive_log_prob.mean()

        return loss


def build_lcl_loss(
    temperature=0.07
):

    return MultiLabelMoCoLoss(
        temperature=temperature
    )
