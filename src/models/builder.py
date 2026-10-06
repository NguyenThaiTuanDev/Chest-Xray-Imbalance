import torch.nn as nn
import torchvision.models as models

def build_model(model_name='densenet201', num_classes=14, pretrained=True):
    """
    Khởi tạo mạng nơ-ron. Mặc định dùng DenseNet201.
    """
    if model_name == 'densenet201':
        # Dùng pre-trained ImageNet weights
        weights = models.DenseNet201_Weights.DEFAULT if pretrained else None
        model = models.densenet201(weights=weights)
        
        # Lớp Classifier gốc của DenseNet201 có đầu vào là 1920 features
        in_features = model.classifier.in_features
        
        # Thay thế bằng lớp Linear 14 chiều cho Multi-label
        model.classifier = nn.Linear(in_features, num_classes)
        
    elif model_name == 'densenet121':
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Kiến trúc {model_name} chưa được cài đặt!")
        
    return model