# Chest-Xray-Imbalance

Nghiên cứu phân loại đa nhãn (14 nhãn/bộ dữ liệu) trên **VinDr-CXR** và **VinDr-PCXR** với DenseNet201 và các hàm mất mát BCE, Weighted BCE, Focal Loss, Asymmetric Loss. Thư mục `src/models/builder.py` cũng chứa kiến trúc học biểu diễn MultiLabelLCL phục vụ giai đoạn tiếp theo.

## Cấu trúc

- `train.py`: huấn luyện baseline / các cấu hình loss.
- `src/data/dataset.py`: 14 nhãn cố định và đối chiếu tên ảnh.
- `src/data/transforms.py`: augmentation train, chuẩn hoá val/test.
- `src/losses/losses.py`: BCE / WBCE / Focal / ASL, và MultiLabelMoCoLoss.
- `src/engine/trainer.py`: huấn luyện và validation.
- `src/engine/evaluation.py`: đánh giá checkpoints, **chọn threshold trên validation**.
- `evaluate.py` và `evaluate_detail.py`: CLI dùng chung evaluator.

## Chuẩn bị dữ liệu

Mỗi dataset `data/processed/<VinDr-CXR|VinDr-PCXR>/` gồm:

```text
train/
test/
labels/
  train_split.csv
  val_split.csv
  test_labels.csv
```

Train/val tham chiếu ảnh trong `train/`; test tham chiếu ảnh trong `test/`. CSV cần `image_id` và đầy đủ 14 cột nhãn đúng tên trong `src/data/dataset.py`.

## Huấn luyện

```bash
python train.py --data_dir data/processed/VinDr-CXR --loss bce --config_name bce --seed 42
python train.py --data_dir data/processed/VinDr-PCXR --loss wbce --wbce_mode 1 --config_name wbce_w1 --seed 42
```

Checkpoint mới lưu tại `outputs/checkpoints/<dataset>/seed<seed>/best_<config_name>_seed<seed>.pth`. Checkpoint cũ không bị tự động sửa.

## Đánh giá đúng nguyên tắc test độc lập

```bash
python evaluate.py --dataset VinDr-CXR
python evaluate.py --dataset VinDr-PCXR
python evaluate_detail.py --dataset VinDr-CXR
```

Hai lệnh đánh giá dùng cùng pipeline, nên mỗi lần sẽ xuất cả:

- `outputs/csv/evaluation_results_<dataset>_val_thresholds.csv`
- `outputs/csv/evaluation_per_class_AP_<dataset>_val_thresholds.csv`

**Chú ý:** Threshold F1 chọn từ validation, cố định trên test. AUC và AP chỉ được tổng hợp cho các nhãn có cả mẫu dương và âm trong test; AP của nhãn không hợp lệ để `NaN`. Các CSV cũ mang hậu tố `all_seeds.csv` được giữ nguyên để đối chiếu, nhưng F1 ở script đánh giá cũ đã tối ưu trên test và không nên dùng làm kết quả cuối.

Việc chọn cấu hình loss/hyperparameters phải dựa trên validation, không chọn sau khi xem test. Trước khi so sánh luận văn cần kiểm tra split trùng ảnh/bệnh nhân, khớp phiên bản mã huấn luyện checkpoints và các bảng kết quả.

## Tái lập

Ghi lại Python/PyTorch/CUDA, số epochs, batch size, accumulation, seed, kiến trúc, augmentation và cấu hình từng loss. Chưa thể xác nhận thực nghiệm tái lập hoàn toàn chỉ dựa vào CSV đã tổng hợp.
