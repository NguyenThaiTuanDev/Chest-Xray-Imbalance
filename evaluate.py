import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from torch.utils.data import DataLoader

# 1. IMPORT TỪ CÁC MODULE SRC
from src.data.dataset import VinDrDataset  # Đã cập nhật tên Class mới
from src.data.transforms import get_transforms
from src.models.builder import build_model

# 2. CÁC HÀM BỔ TRỢ ĐÁNH GIÁ
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
    
    # CHỈ CẦN SỬA THÔNG TIN Ở KHU VỰC NÀY KHI ĐỔI DATASET
    dataset_name = "VinDr-PCXR"  # "VinDr-CXR", "VinDr-PCXR"
    num_classes  = 14            
    
    # Đường dẫn (Tự động thích ứng theo dataset_name)
    test_csv_path  = f"data/processed/{dataset_name}/labels/test_labels.csv"
    test_img_dir   = f"data/processed/{dataset_name}/test"
    
    base_checkpoint_dir = f"outputs/checkpoints/{dataset_name}" 
    output_csv          = f"outputs/evaluation_results_{dataset_name}_all_seeds.csv"

    print(f"[*] Đang xử lý bộ dữ liệu: {dataset_name} | Số nhãn: {num_classes}")

    # --- B. KHỞI TẠO DATASET & DATALOADER ---
    print("[*] Load Dataset & Transforms...")
    test_transform = get_transforms(phase='test')
    
    # Chú ý: Đã truyền thêm tham số dataset_name
    test_dataset = VinDrDataset(
        csv_file=test_csv_path, 
        img_dir=test_img_dir, 
        dataset_name=dataset_name, 
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # --- C. KHỞI TẠO MẠNG (BACKBONE) ---
    print("[*] Build Model...")
    model = build_model(model_name='densenet201', num_classes=num_classes, pretrained=False)
    model = model.to(device)
    
    # --- D. QUÉT CHECKPOINTS XUYÊN CÁC THƯ MỤC SEED ---
    ckpt_paths = glob.glob(os.path.join(base_checkpoint_dir, "**", "*.pth"), recursive=True)
    ckpt_paths.sort() 
    
    if not ckpt_paths:
        print(f"[!] Lỗi: Không tìm thấy file .pth nào trong {base_checkpoint_dir}.")
        return

    print(f"\n[*] Đã tìm thấy {len(ckpt_paths)} checkpoints. Bắt đầu đánh giá...")
    results = []
    
    for ckpt in ckpt_paths:
        ckpt_name = os.path.basename(ckpt)
        seed_folder = os.path.basename(os.path.dirname(ckpt)) 
        
        print(f"\n---> Đang xử lý: [{seed_folder}] {ckpt_name}")
        
        # Load weights
        state_dict = torch.load(ckpt, map_location=device)
        model.load_state_dict(clean_state_dict(state_dict))
        
        # Chạy inference
        y_true, y_prob = evaluate_model(model, test_loader, device)
        
        # TÍNH TOÁN ĐỘ ĐO THÔNG MINH: BỎ QUA CÁC CLASS KHÔNG CÓ BỆNH NHÂN
        roc_aucs = []
        pr_aucs = []
        f1_scores_opt = []
        
        for i in range(num_classes):
            y_t = y_true[:, i]
            y_p = y_prob[:, i]
            
            # Chỉ tính toán nếu bệnh lý này có ít nhất 1 ca dương tính VÀ 1 ca âm tính
            if np.sum(y_t) > 0 and np.sum(y_t) < len(y_t):
                # Tính ROC-AUC và PR-AUC (mAP)
                roc_aucs.append(roc_auc_score(y_t, y_p))
                pr_aucs.append(average_precision_score(y_t, y_p))
                
                # Tìm ngưỡng tối ưu và tính F1-Score
                opt_thresh, _ = find_optimal_threshold(y_t, y_p)
                y_pred_bin = (y_p >= opt_thresh).astype(int)
                f1_scores_opt.append(f1_score(y_t, y_pred_bin, zero_division=0))
            else:
                # Tự động lướt qua các nhãn hiếm bị trống (tránh lỗi đỏ của sklearn)
                pass
                
        # Tính điểm Macro bằng cách trung bình cộng các bệnh hợp lệ
        macro_roc_auc = np.mean(roc_aucs) if roc_aucs else 0.0
        macro_pr_auc = np.mean(pr_aucs) if pr_aucs else 0.0
        macro_f1_opt = np.mean(f1_scores_opt) if f1_scores_opt else 0.0
        
        # Ghi nhận kết quả
        results.append({
            "Seed_Folder": seed_folder,
            "Checkpoint": ckpt_name,
            "Macro_ROC_AUC": round(macro_roc_auc, 4),
            "Macro_PR_AUC_mAP": round(macro_pr_auc, 4),
            "Macro_F1_OptThresh": round(macro_f1_opt, 4)
        })
        
    # --- E. XUẤT KẾT QUẢ CSV ---
    df_results = pd.DataFrame(results)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    
    print(f" ĐÃ HOÀN THÀNH! Kết quả chi tiết lưu tại: {output_csv}")
    print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()