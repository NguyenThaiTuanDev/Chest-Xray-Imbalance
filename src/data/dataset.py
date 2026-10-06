import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

# 14 nhãn bệnh lý chuẩn
TARGET_CLASSES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
    'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
    'Pulmonary fibrosis'
]

class VinDrCXRDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, transform=None):
        df_full = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = TARGET_CLASSES

        # AUTO-FILTER: Chỉ giữ lại những dòng có ảnh thực tế trong thư mục
        df_full['img_path'] = df_full['image_id'].apply(lambda x: os.path.join(self.img_dir, f"{x}.png"))
        self.df = df_full[df_full['img_path'].apply(os.path.exists)].reset_index(drop=True)
        print(f"-> Đã load thành công {len(self.df)} ảnh từ {csv_file}")

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
            
        # Lấy vector nhãn (14 chiều)
        labels = row[self.classes].values.astype('float32')
        labels = torch.tensor(labels)
        
        return image, labels