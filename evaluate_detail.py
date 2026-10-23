import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score
from torch.utils.data import DataLoader

# 1. IMPORT TỪ CÁC MODULE SRC
from src.data.dataset import VinDrDataset
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
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_prob_all.append(probs)
            y_true_all.append(targets.cpu().numpy())
    return np.vstack(y_true_all), np.vstack(y_prob_all)

# 3. KỊCH BẢN CHÍNH
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Khởi động quá trình đánh giá trên thiết bị: {device}")
    
    # CHỈ CẦN ĐỔI TÊN BỘ DỮ LIỆU Ở ĐÂY ("VinDr-CXR" hoặc "VinDr-PCXR")
    dataset_name = "VinDr-CXR" 
    
    test_csv_path  = f"data/processed/{dataset_name}/labels/test_labels.csv"
    test_img_dir   = f"data/processed/{dataset_name}/test"
    base_checkpoint_dir = f"outputs/checkpoints/{dataset_name}"
    
    # File xuất ra chi tiết AP từng nhãn
    output_ap_csv = f"outputs/csv/evaluation_per_class_AP_{dataset_name}_all_seeds.csv"

    # --- A. LẤY TỰ ĐỘNG DANH SÁCH NHÃN CHUẨN XÁC TỪ FILE TEST ---
    df_test_raw = pd.read_csv(test_csv_path)
    meta_cols = {'image_id', 'id', 'split', 'fold', 'study_id', 'series_id', 'patient_id', 'filepath', 'path', 'filename'}
    label_names = [c for c in df_test_raw.columns if c.lower() not in meta_cols and np.issubdtype(df_test_raw[c].dtype, np.number)]
    num_classes = len(label_names)

    print(f"[*] Bộ dữ liệu: {dataset_name} | Số nhãn: {num_classes}")
    print(f"[*] Danh sách nhãn: {label_names}")

    # --- B. KHỞI TẠO DATASET & DATALOADER ---
    print("[*] Load Dataset & Transforms...")
    test_transform = get_transforms(phase='test')
    test_dataset = VinDrDataset(
        csv_file=test_csv_path, 
        img_dir=test_img_dir, 
        dataset_name=dataset_name, 
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # --- C. KHỞI TẠO MẠNG (BACKBONE) ---
    print("[*] Build Model DenseNet-201...")
    model = build_model(model_name='densenet201', num_classes=num_classes, pretrained=False)
    model = model.to(device)
    
    # --- D. QUÉT TOÀN BỘ CHECKPOINTS ---
    ckpt_paths = glob.glob(os.path.join(base_checkpoint_dir, "**", "*.pth"), recursive=True)
    ckpt_paths.sort() 
    
    if not ckpt_paths:
        print(f"[!] Lỗi: Không tìm thấy file .pth nào trong {base_checkpoint_dir}.")
        return

    print(f"\n[*] Đã tìm thấy {len(ckpt_paths)} checkpoints. Bắt đầu đánh giá chi tiết từng nhãn...")
    
    results_ap = []
    
    for ckpt in ckpt_paths:
        ckpt_name = os.path.basename(ckpt)
        seed_folder = os.path.basename(os.path.dirname(ckpt)) 
        
        print(f"---> Đang xử lý: [{seed_folder}] {ckpt_name}")
        
        # Load weights
        state_dict = torch.load(ckpt, map_location=device)
        state_dict = state_dict["model_state_dict"] if "model_state_dict" in state_dict else state_dict
        model.load_state_dict(clean_state_dict(state_dict))
        
        # Chạy inference
        y_true, y_prob = evaluate_model(model, test_loader, device)
        
        # Khởi tạo dòng kết quả với thông tin cơ bản
        row_ap = {
            "Seed": seed_folder,
            "Checkpoint": ckpt_name
        }
        
        pr_aucs = []
        
        # Lặp qua từng nhãn và tính AP
        for i, lbl in enumerate(label_names):
            y_t = y_true[:, i]
            y_p = y_prob[:, i]
            
            if np.sum(y_t) > 0:
                ap = average_precision_score(y_t, y_p)
                pr_aucs.append(ap)
                row_ap[lbl] = round(ap, 4)
            else:
                row_ap[lbl] = 0.0
                
        # Tính điểm Macro mAP trung bình
        macro_map = np.mean(pr_aucs) if pr_aucs else 0.0
        
        # Đưa Macro_PR_AUC_mAP lên cột thứ 3 ngay sau Checkpoint
        row_final = {
            "Seed": seed_folder,
            "Checkpoint": ckpt_name,
            "Macro_PR_AUC_mAP": round(macro_map, 4)
        }
        # Ghép các cột nhãn phía sau
        for lbl in label_names:
            row_final[lbl] = row_ap[lbl]
            
        results_ap.append(row_final)
        
    # --- E. XUẤT RA FILE CSV ---
    df_results = pd.DataFrame(results_ap)
    os.makedirs(os.path.dirname(output_ap_csv), exist_ok=True)
    df_results.to_csv(output_ap_csv, index=False)
    
    print("\n" + "="*80)
    print(f" ĐÃ HOÀN THÀNH! Kết quả chi tiết theo từng nhãn đã lưu tại:\n-> {output_ap_csv}")
    print("="*80)
    print(df_results.head(5).to_markdown(index=False))

if __name__ == "__main__":
    main()