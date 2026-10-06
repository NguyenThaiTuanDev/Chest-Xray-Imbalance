import numpy as np
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score

def find_optimal_thresholds(y_true, y_prob):
    num_classes = y_true.shape[1]
    best_thresholds = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(num_classes):
            # Nếu class đó chỉ toàn 0 hoặc toàn 1, mặc định threshold = 0.5
            if len(np.unique(y_true[:, i])) == 1:
                best_thresholds.append(0.5)
                continue
                
            precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_prob[:, i])
            
            denominator = (precision + recall)
            denominator[denominator == 0] = 1e-6
            f1_scores = 2 * (precision * recall) / denominator
            
            best_idx = np.argmax(f1_scores)
            best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_thresholds.append(best_thresh)
            
    return np.array(best_thresholds)

def evaluate_metrics(y_true, y_prob):
    """Tính toán an toàn, bỏ qua các class không có Positive samples"""
    num_classes = y_true.shape[1]
    auc_list = []
    ap_list = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(num_classes):
            # Chỉ tính AUC và AP nếu class i có cả nhãn 0 và nhãn 1
            if len(np.unique(y_true[:, i])) > 1:
                auc_list.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
                ap_list.append(average_precision_score(y_true[:, i], y_prob[:, i]))
    
    # Tính trung bình (Macro) các class hợp lệ
    macro_auc = np.mean(auc_list) if len(auc_list) > 0 else 0.0
    mAP = np.mean(ap_list) if len(ap_list) > 0 else 0.0

    # F1-Score với Optimal Thresholds
    optimal_thresholds = find_optimal_thresholds(y_true, y_prob)
    y_pred = (y_prob >= optimal_thresholds).astype(int)
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return macro_auc, mAP, f1_macro, optimal_thresholds