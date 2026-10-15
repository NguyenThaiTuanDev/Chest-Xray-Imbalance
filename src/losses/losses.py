import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Baseline BCE
class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)

# 2. Weighted BCE
class WBCELoss(nn.Module):
    def __init__(self, pos_weights):
        """ pos_weights: Tensor 14 chiều """
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        
    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)

# 3. Focal Loss
class FocalLossMultiLabel(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        
        # Cân bằng nhãn với Alpha (Nếu có)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

# 4. Asymmetric Loss (ASL) - ĐÃ ĐƯỢC FIX LỖI ỔN ĐỊNH SỐ HỌC (NUMERICAL STABILITY)
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, margin=0.05, eps=1e-5):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.eps = eps

    def forward(self, logits, targets):
        # 1. Ép về float32 để tính loss an toàn tuyệt đối trong môi trường AMP/autocast
        logits = logits.float()
        targets = targets.float()

        # Trường hợp suy biến đặc biệt: gamma_pos = gamma_neg = margin = 0 -> Dùng thẳng BCE an toàn
        if self.gamma_pos == 0.0 and self.gamma_neg == 0.0 and self.margin == 0.0:
            return F.binary_cross_entropy_with_logits(logits, targets)

        # Tính xác suất qua Sigmoid
        probs_pos = torch.sigmoid(logits)
        probs_neg = 1.0 - probs_pos
        
        # Probability Shifting (Hard thresholding cho mẫu âm tính)
        if self.margin > 0:
            probs_neg = (probs_neg + self.margin).clamp(max=1.0)
            
        # Kẹp xác suất trong khoảng [eps, 1.0 - eps] chống log(0)
        probs_pos = probs_pos.clamp(min=self.eps, max=1.0 - self.eps)
        probs_neg = probs_neg.clamp(min=self.eps, max=1.0 - self.eps)
            
        # Basic Binary Cross Entropy
        loss_pos = targets * torch.log(probs_pos)
        loss_neg = (1.0 - targets) * torch.log(probs_neg)
        
        # Asymmetric Focusing (Tính trọng số mà không làm trôi dạt gradient - chuẩn Alibaba)
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            with torch.no_grad():
                pt0 = probs_pos * targets
                pt1 = probs_neg * (1.0 - targets)
                pt = pt0 + pt1
                
                gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
                weight = torch.pow(1.0 - pt, gamma)
            
            loss_pos = loss_pos * weight
            loss_neg = loss_neg * weight
            
        loss = -(loss_pos + loss_neg)
        return loss.mean()