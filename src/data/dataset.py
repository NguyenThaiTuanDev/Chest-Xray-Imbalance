import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

# 1. NHÃN BỆNH LÝ VINDR-CXR (Người lớn) - 14 nhãn
VINDR_CXR_CLASSES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
    'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
    'Pulmonary fibrosis'
]

# 2. NHÃN BỆNH LÝ VINDR-PCXR (Trẻ em) - 14 nhãn
VINDR_PCXR_CLASSES = [
    'Bronchitis', 'Brocho-pneumonia', 'Other disease', 'Bronchiolitis', 
    'Situs inversus', 'Pneumonia', 'Pleuro-pneumonia', 'Diagphramatic hernia', 
    'Tuberculosis', 'Congenital emphysema', 'CPAM', 'Hyaline membrane disease', 
    'Mediastinal tumor', 'Lung tumor'
]

class VinDrDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, dataset_name: str = 'VinDr-CXR', transform=None):
        """
        dataset_name: 'VinDr-CXR' hoặc 'VinDr-PCXR'
        """
        df_full = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Chọn danh sách nhãn dựa vào tên bộ dữ liệu
        if dataset_name == 'VinDr-CXR':
            self.classes = VINDR_CXR_CLASSES
        elif dataset_name == 'VinDr-PCXR':
            self.classes = VINDR_PCXR_CLASSES
        else:
            # BACKUP LUÔN ĐÚNG: Tự động lấy tất cả các cột trong CSV (trừ cột image_id)
            self.classes = df_full.columns[1:].tolist()
            print(f"⚠️ Cảnh báo: Tên dataset không xác định. Tự động lấy {len(self.classes)} nhãn từ file CSV.")

        # AUTO-FILTER: Chỉ giữ lại những dòng có ảnh thực tế trong thư mục
        df_full['img_path'] = df_full['image_id'].apply(lambda x: os.path.join(self.img_dir, f"{x}.png"))
        self.df = df_full[df_full['img_path'].apply(os.path.exists)].reset_index(drop=True)
        
        print(f"-> Đã load thành công {len(self.df)} ảnh từ {csv_file}")
        print(f"-> Chế độ: {dataset_name} | Số lượng nhãn: {len(self.classes)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['img_path']
        
        # Đọc ảnh bằng OpenCV và chuyển BGR -> RGB
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Lỗi đọc ảnh tại: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Áp dụng Augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        # Lấy vector nhãn (Dùng list comprehension để đảm bảo lấy đúng thứ tự cột)
        labels = row[self.classes].values.astype('float32')
        labels = torch.tensor(labels)
        
        return image, labels