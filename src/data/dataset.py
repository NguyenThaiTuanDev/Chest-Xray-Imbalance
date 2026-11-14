from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

VINDR_CXR_CLASSES = [
    'Aortic enlargement','Atelectasis','Calcification','Cardiomegaly',
    'Consolidation','ILD','Infiltration','Lung Opacity','Nodule/Mass',
    'Other lesion','Pleural effusion','Pleural thickening','Pneumothorax',
    'Pulmonary fibrosis'
]
VINDR_PCXR_CLASSES = [
    'Bronchitis','Brocho-pneumonia','Other disease','Bronchiolitis',
    'Situs inversus','Pneumonia','Pleuro-pneumonia','Diagphramatic hernia',
    'Tuberculosis','Congenital emphysema','CPAM','Hyaline membrane disease',
    'Mediastinal tumor','Lung tumor'
]
CLASSES = {'VinDr-CXR': VINDR_CXR_CLASSES, 'VinDr-PCXR': VINDR_PCXR_CLASSES}
IMAGE_EXTENSIONS = {'.png','.jpg','.jpeg','.bmp'}

class VinDrDataset(Dataset):
    def __init__(self, csv_file, img_dir, dataset_name='VinDr-CXR', transform=None,
                 is_contrastive=False, image_id_col='image_id', label_columns=None,
                 strict_binary=True, strict_images=True):
        self.csv_file = Path(csv_file)
        self.img_dir = Path(img_dir)
        self.dataset_name = dataset_name
        self.transform = transform
        self.is_contrastive = is_contrastive
        self.image_id_col = image_id_col
        if not self.csv_file.is_file():
            raise FileNotFoundError(self.csv_file)
        if not self.img_dir.is_dir():
            raise FileNotFoundError(self.img_dir)
        df = pd.read_csv(self.csv_file, dtype={image_id_col: str})
        if image_id_col not in df:
            raise ValueError(f'Missing ID column: {image_id_col}')
        self.classes = list(label_columns) if label_columns is not None else list(CLASSES[dataset_name])
        if len(self.classes) != 14 or len(set(self.classes)) != 14:
            raise ValueError('Expected 14 unique class names')
        missing = [c for c in self.classes if c not in df]
        if missing:
            raise ValueError(f'Missing label columns: {missing}')
        if df[image_id_col].isna().any() or df[image_id_col].str.strip().eq('').any():
            raise ValueError('Missing image IDs')
        if df[image_id_col].duplicated().any():
            raise ValueError(f'Duplicate image IDs: {df.loc[df[image_id_col].duplicated(), image_id_col].head().tolist()}')
        values = df[self.classes].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite labels')
        if strict_binary and not np.isin(values, [0.,1.]).all():
            raise ValueError('Labels must be binary 0/1')
        if not strict_binary and ((values < 0).any() or (values > 1).any()):
            raise ValueError('Labels outside [0,1]')
        index = {}
        for path in self.img_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                if path.stem in index:
                    raise ValueError(f'Duplicate image stem: {path.stem}')
                index[path.stem] = path
        rows, paths, missing_images = [], [], []
        for i, image_id in enumerate(df[image_id_col]):
            path = index.get(image_id)
            if path is None:
                missing_images.append(image_id)
            else:
                rows.append(i)
                paths.append(path)
        if strict_images and missing_images:
            raise FileNotFoundError(f'{len(missing_images)} images missing, examples: {missing_images[:10]}')
        if not paths:
            raise RuntimeError('No matching images')
        self.df = df.iloc[rows].reset_index(drop=True)
        self.image_paths = paths
        self.labels = torch.from_numpy(values[rows].copy()).float()
        self.missing_images = missing_images
        print(f'{dataset_name}: {len(self)} images, {len(self.classes)} classes, '
              f'{len(missing_images)} missing; positives={self.labels.sum(0).int().tolist()}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f'Unreadable image: {path}')
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError(f'Unsupported image shape {image.shape}: {path}')
        if image.dtype == np.uint16:
            image = (image.astype(np.float32) / 257.).clip(0,255).astype(np.uint8)
        elif image.dtype != np.uint8:
            raise ValueError(f'Unsupported dtype {image.dtype}: {path}')
        if self.transform is None:
            raise ValueError('Transform required')
        label = self.labels[idx]
        if self.is_contrastive:
            return self.transform(image=image)['image'], self.transform(image=image)['image'], label
        return self.transform(image=image)['image'], label
