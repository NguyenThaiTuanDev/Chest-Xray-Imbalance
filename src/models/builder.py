import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class MultiLabelLCL(nn.Module):
    def __init__(self, num_classes=14, proj_dim=128, queue_size=4096,
                 momentum=0.999, pretrained=True):
        super().__init__()
        if min(num_classes, proj_dim, queue_size) < 1:
            raise ValueError('Dimensions must be positive')
        if not 0 <= momentum < 1:
            raise ValueError('momentum must be in [0,1)')
        self.num_classes, self.proj_dim = num_classes, proj_dim
        self.queue_size, self.momentum = queue_size, momentum
        weights = models.DenseNet201_Weights.DEFAULT if pretrained else None
        backbone_q = models.densenet201(weights=weights)
        dim = backbone_q.classifier.in_features
        self.encoder_q = backbone_q.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(dim, num_classes)
        self.projector_q = nn.Sequential(nn.Linear(dim,2048), nn.ReLU(inplace=True), nn.Linear(2048,proj_dim))
        self.encoder_k = models.densenet201(weights=None).features
        self.projector_k = nn.Sequential(nn.Linear(dim,2048), nn.ReLU(inplace=True), nn.Linear(2048,proj_dim))
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        self.projector_k.load_state_dict(self.projector_q.state_dict())
        self.encoder_k.requires_grad_(False)
        self.projector_k.requires_grad_(False)
        self.register_buffer('queue_features', torch.zeros(proj_dim,queue_size))
        self.register_buffer('queue_labels', torch.zeros(num_classes,queue_size))
        self.register_buffer('queue_ptr', torch.zeros(1,dtype=torch.long))
        self.register_buffer('queue_count', torch.zeros(1,dtype=torch.long))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.encoder_k.eval()
        self.projector_k.eval()
        return self

    @torch.no_grad()
    def update_momentum_encoder(self):
        m = self.momentum
        for q,k in ((self.encoder_q,self.encoder_k),(self.projector_q,self.projector_k)):
            for p_q,p_k in zip(q.parameters(),k.parameters()):
                p_k.lerp_(p_q,1-m)
            for b_q,b_k in zip(q.buffers(),k.buffers()):
                b_k.copy_(b_q)

    @torch.no_grad()
    def get_valid_queue(self):
        n = int(self.queue_count.item())
        return self.queue_features[:,:n].detach().clone(), self.queue_labels[:,:n].detach().clone()

    @torch.no_grad()
    def reset_queue(self):
        self.queue_features.zero_(); self.queue_labels.zero_()
        self.queue_ptr.zero_(); self.queue_count.zero_()

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, labels):
        if keys.ndim != 2 or keys.shape[1] != self.proj_dim:
            raise ValueError('keys shape must be [B,proj_dim]')
        if labels.ndim != 2 or labels.shape != (keys.shape[0],self.num_classes):
            raise ValueError('labels shape must be [B,num_classes]')
        if not torch.isfinite(keys).all() or not torch.isfinite(labels).all():
            raise ValueError('Nonfinite queue input')
        if not keys.shape[0]:
            return
        keys = F.normalize(keys.detach().float(),dim=1).to(self.queue_features)
        labels = labels.detach().to(self.queue_labels)
        k = self.queue_size
        original = keys.shape[0]
        ptr = (int(self.queue_ptr.item()) + max(0,original-k)) % k
        keys, labels = keys[-k:], labels[-k:]
        b = keys.shape[0]
        first = min(b,k-ptr)
        self.queue_features[:,ptr:ptr+first] = keys[:first].T
        self.queue_labels[:,ptr:ptr+first] = labels[:first].T
        if b > first:
            self.queue_features[:,:b-first] = keys[first:].T
            self.queue_labels[:,:b-first] = labels[first:].T
        self.queue_ptr[0] = (ptr+b)%k
        self.queue_count[0] = min(k,int(self.queue_count.item())+original)

    def extract_features(self, images):
        return self.pool(F.relu(self.encoder_q(images),inplace=False)).flatten(1)

    def forward(self, im_q, im_k=None):
        pooled_q = self.extract_features(im_q)
        logits = self.classifier(pooled_q)
        if im_k is None or not self.training:
            return logits
        q = F.normalize(self.projector_q(pooled_q).float(),dim=1)
        with torch.no_grad():
            pooled_k = self.pool(F.relu(self.encoder_k(im_k),inplace=False)).flatten(1)
            k = F.normalize(self.projector_k(pooled_k).float(),dim=1)
        return logits,q,k

def build_model(model_name='densenet201',num_classes=14,pretrained=True,**kwargs):
    if model_name == 'densenet201':
        weights = models.DenseNet201_Weights.DEFAULT if pretrained else None
        model = models.densenet201(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features,num_classes)
        return model
    if model_name == 'lcl_densenet201':
        return MultiLabelLCL(num_classes=num_classes,pretrained=pretrained,
                             proj_dim=kwargs.get('proj_dim',128),
                             queue_size=kwargs.get('queue_size',4096),
                             momentum=kwargs.get('momentum',0.999))
    raise ValueError(f'Unsupported model: {model_name}')
