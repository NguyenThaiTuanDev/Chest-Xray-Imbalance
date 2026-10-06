import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_transforms(phase: str):
    """
    Trả về pipeline augmentation dựa trên phase (train/val/test).
    """
    if phase == 'train':
        return A.Compose([
            # Cập nhật chuẩn Albumentations mới: dùng border_mode và fill
            A.Affine(
                scale=(1.0, 1.0), 
                translate_percent=(0.0, 0.0), 
                rotate=(-5, 5),
                border_mode=cv2.BORDER_CONSTANT, 
                fill=0, 
                p=0.5
            ),
            
            # Thay đổi độ sáng/tương phản ngẫu nhiên (±10%)
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=0.5
            ),
            
            # Chuẩn hóa theo chuẩn ImageNet
            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
        
    elif phase in ['val', 'test']:
        return A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    else:
        raise ValueError("Phase chỉ có thể là 'train', 'val' hoặc 'test'")