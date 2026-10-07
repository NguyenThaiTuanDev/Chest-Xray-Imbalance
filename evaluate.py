import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import VinDrCXRDataset
from src.data.transforms import get_transforms
from src.models.builder import build_model
from src.engine.metrics import evaluate_metrics, find_optimal_thresholds

@torch.no_grad()
def get_predictions(model, dataloader, device):
    model.eval()
    all_targets = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc="Predicting")
    for images, targets in pbar:
        images = images.to(device)
        # Bật autocast cho lúc inference để chạy nhanh hơn
        with torch.amp.autocast('cuda'):
            logits = model(images)
            probs = torch.sigmoid(logits)
            
        all_targets.append(targets.cpu())
        all_probs.append(probs.cpu())
        
    return torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Đang chạy Evaluation trên thiết bị: {device}")
    
    # 1. LOAD DATASETS (Val để tìm Threshold, Test để đánh giá)
    val_dataset = VinDrCXRDataset(
        csv_file=os.path.join(args.data_dir, "labels", "val_split.csv"),
        img_dir=os.path.join(args.data_dir, "images_train_512"),
        transform=get_transforms('val')
    )
    test_dataset = VinDrCXRDataset(
        csv_file=os.path.join(args.data_dir, "labels", "test_labels.csv"),
        img_dir=os.path.join(args.data_dir, "images_test_512"),
        transform=get_transforms('test')
    )
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 2. LOAD MODEL & TRỌNG SỐ
    model = build_model(model_name='densenet201', num_classes=14, pretrained=False)
    
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Không tìm thấy file weights tại {args.weights}")
        
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    print(f"✅ Đã load trọng số thành công từ: {args.weights}")
    
    # 3. TÌM NGƯỠNG TỐI ƯU TRÊN TẬP VALIDATION
    print("\nBước 1: Tìm Ngưỡng tối ưu trên tập Validation...")
    y_val_true, y_val_prob = get_predictions(model, val_loader, device)
    optimal_thresholds = find_optimal_thresholds(y_val_true, y_val_prob)
    
    # 4. ĐÁNH GIÁ TRÊN TẬP TEST
    print("\nBước 2: Dự đoán và Đánh giá trên tập TEST...")
    y_test_true, y_test_prob = get_predictions(model, test_loader, device)
    
    test_auc, test_map, test_f1, _ = evaluate_metrics(y_test_true, y_test_prob, thresholds=optimal_thresholds)
    
    print("\n" + "="*50)
    print(f"KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (3000 ảnh) - MODEL: {args.weights.split('/')[-1]}")
    print("="*50)
    print(f"🔥 Macro-AUC : {test_auc:.4f}")
    print(f"🔥 mAP (PR-AUC): {test_map:.4f}")
    print(f"🔥 Macro-F1  : {test_f1:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Đường dẫn đến thư mục processed")
    parser.add_argument('--weights', type=str, required=True, help="Đường dẫn file .pth cần test")
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    main(args)