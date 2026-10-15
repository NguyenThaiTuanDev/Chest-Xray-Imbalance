import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler 

# IMPORT CUSTOM MODULES CỦA DỰ ÁN
from src.data.dataset import VinDrCXRDataset
from src.data.transforms import get_transforms
from src.models.builder import build_model
from src.losses.losses import BCELoss, WBCELoss, FocalLossMultiLabel, AsymmetricLoss
from src.engine.trainer import train_one_epoch, validate

# CỐ ĐỊNH SEED TÁI LẬP (REPRODUCIBILITY)
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

# TỰ ĐỘNG NHẬN DIỆN & TỐI ƯU HÓA GPU
def setup_hardware():
    """
    Tự động quét phần cứng (GPU/CPU), đo VRAM và bật cờ tăng tốc phù hợp.
    """
    print("🖥️  HỆ THỐNG ĐANG KIỂM TRA & NHẬN DIỆN PHẦN CỨNG...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpus = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        major, minor = torch.cuda.get_device_capability(0)
        
        print(f"✅ Phát hiện: {num_gpus} x GPU CUDA")
        print(f"🎮 Dòng GPU đang chạy: {gpu_name}")
        print(f"💾 Dung lượng VRAM khả dụng: {total_vram:.2f} GB")
        print(f"⚙️  Kiến trúc vi xử lý (Capability): {major}.{minor}")
        
        # Tự động kích hoạt TF32 cho kiến trúc Ampere (A100) và Ada Lovelace (L4)
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("🚀 [TỐI ƯU CAO CẤP]: Đã bật TensorFloat-32 (TF32) tăng tốc Tensor Cores!")
        else:
            print("⚡ [TỐI ƯU CƠ BẢN]: Bật Mixed Precision FP16 chuẩn cho kiến trúc Turing/Volta (T4/V100).")
            
        if num_gpus > 1:
            print(f"🔥 Bật chế độ DataParallel chia tải trên cả {num_gpus} GPUs!")
    else:
        device = torch.device('cpu')
        num_gpus = 0
        print("⚠️ Không tìm thấy GPU, hệ thống sẽ sử dụng CPU.")

    # Tự động tính num_workers tránh cảnh báo nghẽn CPU
    cpu_cores = os.cpu_count() or 2
    optimal_workers = max(1, min(4, cpu_cores // 2 if cpu_cores > 2 else 2))
    print(f"🧠 CPU khả dụng: {cpu_cores} Cores -> Tự động cấu hình num_workers = {optimal_workers}")
    
    return device, num_gpus, optimal_workers

# TÍNH TOÁN TRỌNG SỐ CHO WBCE
def calculate_pos_weights(df, mode):
    """
    Tính trọng số cho các nhãn bệnh lý.
    Mode 0: Suy biến về BCE (Trọng số w = 1.0 cho tất cả các lớp).
    """
    target_classes = df.columns[1:15]
    pos_counts = df[target_classes].sum().values
    neg_counts = len(df) - pos_counts
    
    pos_counts[pos_counts == 0] = 1 # Tránh chia cho 0
    ratio = neg_counts / pos_counts
    
    if mode == 1:
        w = np.sqrt(ratio)
        print("   -> WBCE Mode 1: Trọng số căn bậc hai (sqrt ratio)")
    elif mode == 2:
        w = np.clip(ratio, a_min=None, a_max=10.0)
        print("   -> WBCE Mode 2: Trọng số chặn trần tối đa 10.0")
    elif mode == 3:
        w = np.clip(ratio, a_min=None, a_max=50.0)
        print("   -> WBCE Mode 3: Trọng số chặn trần tối đa 50.0")
    else:
        # MODE 0: TRƯỜNG HỢP SUY BIẾN VỀ BCE
        w = np.ones_like(ratio)
        print("   -> [SANITY CHECK] WBCE Mode 0: Vector trọng số w = 1.0 (Suy biến về BCE gốc)")
        
    return torch.tensor(w, dtype=torch.float32)

# CHƯƠNG TRÌNH HUẤN LUYỆN CHÍNH
def main(args):
    # Khởi tạo Seed và Thiết bị
    seed_everything(args.seed)
    device, num_gpus, optimal_workers = setup_hardware()
    
    effective_batch = args.batch_size * args.accum_steps
    print(f"📊 Cấu hình Batch: Batch={args.batch_size}, Tích lũy Gradient={args.accum_steps} -> Kích thước thực = {effective_batch}")
    
    # Load Datasets
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
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=optimal_workers, 
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size * 2, 
        shuffle=False, 
        num_workers=optimal_workers, 
        pin_memory=(device.type == 'cuda')
    )
    
    # Khởi tạo Model
    model = build_model(model_name='densenet201', num_classes=14, pretrained=True)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # THIẾT LẬP HÀM MẤT MÁT VÀ KIỂM TRA SUY BIẾN (SANITY CHECK)
    if args.loss == 'bce':
        criterion = BCELoss()
        print("🎯 Hàm Mất mát: BCE (Baseline Chuẩn)")
        
    elif args.loss == 'wbce':
        pos_weights = calculate_pos_weights(train_dataset.df, args.wbce_mode).to(device)
        criterion = WBCELoss(pos_weights=pos_weights)
        if args.wbce_mode == 0:
            print("   ⚠️  CẢNH BÁO SANITY CHECK: WBCE đang được đưa về phương trình BCE gốc!")
            
    elif args.loss == 'focal':
        criterion = FocalLossMultiLabel(gamma=args.gamma, alpha=args.alpha)
        alpha_str = f"alpha={args.alpha}" if args.alpha is not None else "alpha=None"
        print(f"🎯 Hàm Mất mát: Focal Loss (gamma={args.gamma}, {alpha_str})")
        
        # Kiểm tra suy biến Focal -> BCE
        if args.gamma == 0.0 and args.alpha is None:
            print("   ⚠️  CẢNH BÁO SANITY CHECK: gamma=0.0 và alpha=None -> Focal Loss suy biến hoàn toàn về BCE!")
            
    elif args.loss == 'asl':
        criterion = AsymmetricLoss(gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg, margin=args.margin)
        print(f"🎯 Hàm Mất mát: ASL (gamma_+={args.gamma_pos}, gamma_-={args.gamma_neg}, margin={args.margin})")
        
        # Kiểm tra suy biến ASL -> BCE
        if args.gamma_pos == 0.0 and args.gamma_neg == 0.0 and args.margin == 0.0:
            print("   ⚠️  CẢNH BÁO SANITY CHECK: gamma_+=0, gamma_-=0, margin=0.0 -> ASL suy biến hoàn toàn về BCE!")
        
    # Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda')) 
    
    # Training Loop
    best_map = 0.0
    os.makedirs('outputs/checkpoints', exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args.accum_steps)
        val_loss, val_auc, val_map, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Macro-AUC:  {val_auc:.4f} | mAP: {val_map:.4f} | Macro-F1: {val_f1:.4f}")
        
        # Lưu file weights theo tên config_name để không bị đè file
        if val_map > best_map:
            best_map = val_map
            save_path = f"outputs/checkpoints/best_{args.config_name}_seed{args.seed}.pth"
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), save_path)
            print(f"   🔥 Đã lưu checkpoint tốt nhất: {save_path} (mAP: {best_map:.4f})")

# CẤU HÌNH THAM SỐ DÒNG LỆNH (ARGPARSE)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Phân loại Đa nhãn X-Quang Ngực")
    
    # Cấu hình chung
    parser.add_argument('--data_dir', type=str, default=r"data/processed")
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'wbce', 'focal', 'asl'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--accum_steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config_name', type=str, default='bce', help="Tên cấu hình để lưu file checkpoint không bị ghi đè")
    
    # Cấu hình WBCE (Có mode 0 cho Sanity Check)
    parser.add_argument('--wbce_mode', type=int, default=1, choices=[0, 1, 2, 3])
    
    # Cấu hình Focal Loss
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=None)
    
    # Cấu hình Asymmetric Loss (ASL)
    parser.add_argument('--gamma_pos', type=float, default=0.0)
    parser.add_argument('--gamma_neg', type=float, default=4.0)
    parser.add_argument('--margin', type=float, default=0.05)
    
    args = parser.parse_args()
    main(args)