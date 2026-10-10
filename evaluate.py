import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from torch.utils.data import DataLoader

# 1. IMPORT TỪ CÁC MODULE BÊN TRONG THƯ MỤC SRC
from src.data.dataset import VinDrCXRDataset
from src.data.transforms import get_transforms
from src.models.builder import build_model

# 2. CÁC HÀM BỔ TRỢ CHO QUÁ TRÌNH EVALUATE
def clean_state_dict(state_dict):
    """Xóa tiền tố 'module.' nếu model được train bằng nn.DataParallel"""
    return {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict.items()}

def find_optimal_threshold(y_true, y_prob):
    """Tìm ngưỡng (threshold) mang lại F1-Score cao nhất cho 1 class"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    best_f1, best_thresh = 0.0, 0.5
    
    for precision, recall, threshold in zip(precisions, recalls, thresholds):
        if precision + recall == 0:
            continue
        f1 = 2 * (precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1, best_thresh = f1, threshold
            
    return best_thresh, best_f1

def evaluate_model(model, dataloader, device):
    """Tiến hành inference trên toàn bộ tập Test"""
    model.eval()
    y_true_all, y_prob_all = [], []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs) 
            probs = torch.sigmoid(outputs).cpu().numpy() # Multi-label Sigmoid
            
            y_prob_all.append(probs)
            y_true_all.append(targets.cpu().numpy())
            
    return np.vstack(y_true_all), np.vstack(y_prob_all)

# 3. KỊCH BẢN CHÍNH
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Khởi động quá trình đánh giá trên thiết bị: {device}")
    
    # --- A. ĐƯỜNG DẪN CONFIG ---
    test_csv_path  = "data/processed/labels/test_labels.csv"
    test_img_dir   = "data/processed/images_test_512"
    checkpoint_dir = "outputs/checkpoints"
    output_csv     = "outputs/evaluation_results.csv"

    # --- B. KHỞI TẠO DATASET & DATALOADER ---
    print("[*] Load Dataset & Transforms...")
    # Gọi hàm từ transforms.py
    test_transform = get_transforms(phase='test')
    
    # Gọi class từ dataset.py
    test_dataset = VinDrCXRDataset(
        csv_file=test_csv_path, 
        img_dir=test_img_dir, 
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # --- C. KHỞI TẠO MẠNG (BACKBONE) ---
    print("[*] Build Model...")
    model = build_model(model_name='densenet201', num_classes=14, pretrained=False)
    model = model.to(device)
    
    # --- D. QUÉT CHECKPOINTS & CHẠY EVALUATE ---
    ckpt_paths = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    ckpt_paths.sort() 
    
    if not ckpt_paths:
        print(f"[!] Không tìm thấy checkpoint nào tại {checkpoint_dir}")
        return

    results = []
    
    for ckpt in ckpt_paths:
        ckpt_name = os.path.basename(ckpt)
        print(f"\n---> Đang xử lý Checkpoint: {ckpt_name}")
        
        # Load weights
        state_dict = torch.load(ckpt, map_location=device)
        model.load_state_dict(clean_state_dict(state_dict))
        
        # Chạy inference
        y_true, y_prob = evaluate_model(model, test_loader, device)
        
        # Tính ROC-AUC và PR-AUC
        try:
            macro_roc_auc = roc_auc_score(y_true, y_prob, average='macro')
            macro_pr_auc = average_precision_score(y_true, y_prob, average='macro') 
        except ValueError:
            macro_roc_auc, macro_pr_auc = 0, 0
            
        # Tìm ngưỡng tối ưu cho 14 bệnh lý
        f1_scores_opt = []
        for i in range(14):
            if np.sum(y_true[:, i]) == 0: continue # Bỏ qua nếu class ko có trong test set
            opt_thresh, _ = find_optimal_threshold(y_true[:, i], y_prob[:, i])
            y_pred_bin = (y_prob[:, i] >= opt_thresh).astype(int)
            f1_scores_opt.append(f1_score(y_true[:, i], y_pred_bin, zero_division=0))
            
        macro_f1_opt = np.mean(f1_scores_opt) if f1_scores_opt else 0
        
        # Ghi nhận kết quả
        results.append({
            "Checkpoint": ckpt_name,
            "Macro_ROC_AUC": round(macro_roc_auc, 4),
            "Macro_PR_AUC_mAP": round(macro_pr_auc, 4),
            "Macro_F1_OptThresh": round(macro_f1_opt, 4)
        })
        
    # --- E. XUẤT KẾT QUẢ CSV ---
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f" ĐÃ HOÀN THÀNH! Kết quả chi tiết lưu tại: {output_csv}")
    print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()