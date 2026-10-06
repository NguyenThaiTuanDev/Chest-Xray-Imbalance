import pandas as pd
import os
from pathlib import Path

def check_alignment(csv_paths, img_dir, split_name):
    print(f"\n{'='*20} KIỂM TRA TẬP {split_name.upper()} {'='*20}")
    
    # 1. Lấy ID từ (các) file CSV
    csv_ids = set()
    # Nếu truyền vào 1 chuỗi/Path thì chuyển thành list để dễ vòng lặp
    if not isinstance(csv_paths, list):
        csv_paths = [csv_paths]
        
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"[LỖI] Không tìm thấy file CSV: {csv_path}")
            return
        df = pd.read_csv(csv_path)
        # Ép kiểu string để so sánh an toàn
        csv_ids.update(df['image_id'].astype(str).tolist())
        
    print(f"-> Tổng số image_id trong (các) CSV: {len(csv_ids)}")

    # 2. Lấy ID từ thư mục ảnh vật lý
    if not os.path.exists(img_dir):
        print(f"[LỖI] Không tìm thấy thư mục ảnh: {img_dir}")
        return
        
    # Duyệt thư mục, lấy tên file bỏ đuôi .png (vd: 'abc.png' -> 'abc')
    img_ids = set([Path(f).stem for f in os.listdir(img_dir) if f.endswith('.png')])
    print(f"-> Tổng số ảnh thực tế trong thư mục: {len(img_ids)}")

    # 3. Đối chiếu bằng Set (Tập hợp)
    missing_images = csv_ids - img_ids # Có trong CSV nhưng mất ảnh
    extra_images = img_ids - csv_ids   # Có ảnh nhưng không có trong CSV

    # 4. In kết quả
    if len(missing_images) == 0 and len(extra_images) == 0:
        print("✅ HOÀN HẢO: Dữ liệu ảnh và CSV khớp nhau 100%!")
    else:
        if len(missing_images) > 0:
            print(f"❌ LỖI NGHIÊM TRỌNG: Có {len(missing_images)} ID trong CSV nhưng KHÔNG CÓ ảnh.")
            print(f"   Ví dụ vài ID bị thiếu: {list(missing_images)[:3]}")
            print("   => Dataloader sẽ bị crash nếu đem đi train!")
            
        if len(extra_images) > 0:
            print(f"⚠️ LƯU Ý: Có {len(extra_images)} ảnh rác trong thư mục (Không có nhãn trong CSV).")

if __name__ == "__main__":
    base_path = Path(".")
    
    # 1. KIỂM TRA TẬP TRAIN & VAL
    # Kết hợp ID của cả train_split và val_split để so sánh với thư mục images_train_512
    train_val_csvs = [
        base_path / "data/processed/labels/train_split.csv",
        base_path / "data/processed/labels/val_split.csv"
    ]
    train_img_dir = base_path / "data/processed/images_train_512"
    
    check_alignment(train_val_csvs, train_img_dir, "Train & Validation")
    
    # 2. KIỂM TRA TẬP TEST
    test_csv = base_path / "data/processed/labels/test_labels.csv"
    test_img_dir = base_path / "data/processed/images_test_512"
    
    check_alignment(test_csv, test_img_dir, "Test")