# 🫁 Chest X-Ray Anomaly Detection: Tackling Multi-Label Imbalance via Loss Function Optimization
**Nghiên cứu học biểu diễn đặc trưng thị giác và tinh chỉnh Hàm mất mát nhằm nâng cao hiệu quả phân lớp nhãn hiếm trên dữ liệu VinDr-CXR.**

---

## 📌 Tổng quan (Overview)
Dự án này tập trung giải quyết thách thức cốt lõi trong chẩn đoán ảnh X-quang ngực (Chest X-ray): **Phân loại đa nhãn (Multi-label Classification)** trên tập dữ liệu có phân phối **mất cân bằng cực độ (Extreme Long-tailed Distribution)**. 

Sử dụng bộ dữ liệu **VinBigData Chest X-ray (VinDr-CXR)**, thay vì làm phức tạp hóa kiến trúc mạng nơ-ron (Architecture-level) vốn đòi hỏi chi phí tính toán khổng lồ, nghiên cứu này chứng minh tính hiệu quả vượt trội của việc **Tối ưu hóa Không gian Hàm mất mát (Loss-level Optimization)**.

### 🚀 Các đóng góp chính (Key Contributions)
- **Mạng trích xuất đặc trưng:** Cài đặt và tinh chỉnh `DenseNet201` (Tận dụng Dense Connectivity để bảo toàn đặc trưng bệnh lý nhỏ).
- **Ablation Study các Hàm Mất Mát:** Đánh giá đối chứng toàn diện giữa:
  - `BCE Loss` (Đường cơ sở - Baseline)
  - `Weighted BCE Loss` (Bù đắp trọng số - Static Weighting)
  - `Focal Loss` (Khai phá mẫu khó - Hard Example Mining)
  - `Asymmetric Loss - ASL` (Triệt tiêu bất đối xứng đa nhãn)
- **Đánh giá chuẩn Y tế:** Sử dụng `PR-AUC (mAP)` và dò tìm `Optimal Threshold` cho F1-Score thay vì ngưỡng 0.5 mặc định.
- **Explainable AI (XAI):** Trực quan hóa bằng `Grad-CAM`, chứng minh các hàm mất mát tiên tiến (ASL/Focal) giúp khử nhiễu nền (background bias) và khoanh vùng chính xác tổn thương.
- **Phương pháp luận:** Mọi thực nghiệm được báo cáo bằng `Mean ± Std` trên 3 random seeds kinh điển `[42, 123, 3407]` nhằm đảm bảo tính minh bạch và tránh P-Hacking.