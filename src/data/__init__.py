import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

TARGET_CLASSES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
    'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
    'Pulmonary fibrosis'
]

class VinDrCXRDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, transform=None):
        """
        Khởi tạo Dataset.
        """
        # 1. Đọc file CSV gốc
        df_full = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = TARGET_CLASSES

        # 2. AUTO-FILTER (Lọc tự động ảnh tồn tại)
        # Bước này cực kỳ hữu ích khi test với 100 ảnh trên Laptop
        # Nó kiểm tra file vật lý có trên đĩa cứng không, nếu có mới giữ lại trong DataFrame
        print(f"Đang kiểm tra dữ liệu từ {csv_file}...")
        df_full['img_path'] = df_full['image_id'].apply(lambda x: os.path.join(self.img_dir, f"{x}.png"))
        
        # Chỉ giữ lại các dòng mà đường dẫn file thực sự tồn tại
        self.df = df_full[df_full['img_path'].apply(os.path.exists)].reset_index(drop=True)
        
        print(f"-> Đã load thành công {len(self.df)} ảnh hợp lệ!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Đã lấy sẵn đường dẫn từ bước init
        img_path = row['img_path'] 
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        labels = row[self.classes].values.astype('float32')
        labels = torch.tensor(labels)
        
        return image, labels