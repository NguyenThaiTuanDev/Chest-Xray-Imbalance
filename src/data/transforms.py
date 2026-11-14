import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_transforms(phase='train', image_size=512):
    if phase not in ('train','val','test'):
        raise ValueError(f'Unsupported phase: {phase}')
    transforms = [A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR)]
    if phase == 'train':
        transforms.extend([
            A.Affine(scale=(0.98,1.02), translate_percent=(-0.01,0.01),
                     rotate=(-5,5), interpolation=cv2.INTER_LINEAR,
                     border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ])
    transforms.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.),
        ToTensorV2(),
    ])
    return A.Compose(transforms)
