import torch
from tqdm import tqdm
from torch.amp import autocast
import numpy as np
from src.engine.metrics import evaluate_metrics

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, accumulation_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad() 
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Mixed Precision Autocast
        with autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, targets)
            loss = loss / accumulation_steps 
            
        scaler.scale(loss).backward()
        
        # Cập nhật Gradient theo accumulation_steps
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() 
            
        running_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f"{running_loss / (step + 1):.4f}"})
        
    return running_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, targets)
            probs = torch.sigmoid(logits)
            
        running_loss += loss.item()
        all_targets.append(targets.cpu())
        all_probs.append(probs.cpu())
        
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    # Bảo hiểm an toàn chống NaN cho Validation
    if np.isnan(all_probs).any():
        all_probs = np.nan_to_num(all_probs, nan=0.5)

    val_loss = running_loss / len(dataloader)
    val_auc, val_map, val_f1, _ = evaluate_metrics(all_targets, all_probs)
    
    return val_loss, val_auc, val_map, val_f1