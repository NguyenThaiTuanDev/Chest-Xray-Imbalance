import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler 

# Import custom modules
from src.data.dataset import VinDrCXRDataset
from src.data.transforms import get_transforms
from src.models.builder import build_model
from src.losses.losses import BCELoss, WBCELoss, FocalLossMultiLabel, AsymmetricLoss
from src.engine.trainer import train_one_epoch, validate

def seed_everything(seed=132):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def calculate_pos_weights(df, mode):
    target_classes = df.columns[1:15]
    pos_counts = df[target_classes].sum().values
    neg_counts = len(df) - pos_counts
    
    pos_counts[pos_counts == 0] = 1 
    ratio = neg_counts / pos_counts
    
    if mode == 1:
        w = np.sqrt(ratio)
    elif mode == 2:
        w = np.clip(ratio, a_min=None, a_max=10.0)
    elif mode == 3:
        w = np.clip(ratio, a_min=None, a_max=50.0)
    else:
        w = np.ones_like(ratio)
        
    return torch.tensor(w, dtype=torch.float32)

def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n🚀 Khởi chạy trên thiết bị: {device.type.upper()}")
    
    # ==========================================
    # KHAI THÁC MULTI-GPU (T4x2) TRÊN KAGGLE
    # ==========================================
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"🔥 TUYỆT VỜI! Phát hiện {num_gpus} GPUs. Đang bật chế độ DataParallel...")
    
    print(f"📊 Cấu hình: Batch={args.batch_size}, Accumulate={args.accum_steps} -> Thực tế={args.batch_size * args.accum_steps}")
    
    # 1. DATASETS & DATALOADERS
    train_dataset = VinDrCXRDataset(
        csv_file=os.path.join(args.data_dir, "labels", "train_split.csv"),
        img_dir=os.path.join(args.data_dir, "images_train_512"),
        transform=get_transforms('train')
    )
    
    val_dataset = VinDrCXRDataset(
        csv_file=os.path.join(args.data_dir, "labels", "val_split.csv"), 
        img_dir=os.path.join(args.data_dir, "images_train_512"),
        transform=get_transforms('val')
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. MODEL & DATAPARALLEL
    model = build_model(model_name='densenet201', num_classes=14, pretrained=True)
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # 3. LOSS FUNCTION
    if args.loss == 'bce':
        criterion = BCELoss()
        print("🎯 Hàm Mất mát: BCE (Baseline)")
    elif args.loss == 'wbce':
        pos_weights = calculate_pos_weights(train_dataset.df, args.wbce_mode).to(device)
        criterion = WBCELoss(pos_weights=pos_weights)
        print(f"🎯 Hàm Mất mát: WBCE (Mode W{args.wbce_mode})")
    elif args.loss == 'focal':
        criterion = FocalLossMultiLabel(gamma=args.gamma, alpha=args.alpha)
        print(f"🎯 Hàm Mất mát: Focal Loss (gamma={args.gamma}, alpha={args.alpha})")
    elif args.loss == 'asl':
        criterion = AsymmetricLoss(gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg, margin=args.margin)
        print(f"🎯 Hàm Mất mát: ASL (gamma_+={args.gamma_pos}, gamma_-={args.gamma_neg}, margin={args.margin})")
        
    # 4. OPTIMIZER & SCALER
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler('cuda') 
    
    # 5. TRAINING LOOP
    best_map = 0.0
    os.makedirs('outputs/checkpoints', exist_ok=True)
    
    print("\n" + "="*50)
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args.accum_steps)
        val_loss, val_auc, val_map, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Macro-AUC:  {val_auc:.4f} | mAP: {val_map:.4f} | Macro-F1: {val_f1:.4f}")
        
        if val_map > best_map:
            best_map = val_map
            save_path = f"outputs/checkpoints/best_{args.config_name}_seed{args.seed}.pth"
            
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), save_path)
            print(f"   🔥 Đã lưu model tốt nhất (mAP: {best_map:.4f}) vào {save_path}")
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r"data/processed")
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'wbce', 'focal', 'asl'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wbce_mode', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--gamma_pos', type=float, default=0.0)
    parser.add_argument('--gamma_neg', type=float, default=4.0)
    parser.add_argument('--margin', type=float, default=0.05)
    
    parser.add_argument('--config_name', type=str, default='bce')
    
    args = parser.parse_args()
    main(args)