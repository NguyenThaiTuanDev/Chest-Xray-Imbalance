"""Task04A integration test with synthetic data, no images/weights needed.

Tests real trainer functions using a tiny API-compatible model, then optionally
runs a 64px DenseNet201 full step on CPU or CUDA via --real-model.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def check(ok, message, results):
    status='PASS' if bool(ok) else 'FAIL'
    print(f'  [{status}] {message}')
    results.append({'check':message,'status':status})
    if status=='FAIL':
        raise AssertionError(message)


class TinyMomentumModel(nn.Module):
    """Fast faithful interface to verify trainer/queue/EMA orchestration."""
    def __init__(self, nclass=14, dim=16, queue_size=7, momentum=0.9):
        super().__init__()
        self.encoder_q=nn.Sequential(nn.Flatten(),nn.Linear(3*16*16,dim))
        self.encoder_k=nn.Sequential(nn.Flatten(),nn.Linear(3*16*16,dim))
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        self.encoder_k.requires_grad_(False)
        self.projector_q=nn.Linear(dim,dim)
        self.projector_k=nn.Linear(dim,dim)
        self.projector_k.load_state_dict(self.projector_q.state_dict())
        self.projector_k.requires_grad_(False)
        self.classifier=nn.Linear(dim,nclass)
        self.queue_size=queue_size
        self.momentum=momentum
        self.register_buffer('queue_features',torch.zeros(dim,queue_size))
        self.register_buffer('queue_labels',torch.zeros(nclass,queue_size))
        self.register_buffer('queue_ptr',torch.zeros(1,dtype=torch.long))
        self.register_buffer('queue_count',torch.zeros(1,dtype=torch.long))
        self.ema_calls=0
    def train(self,mode=True):
        super().train(mode)
        self.encoder_k.eval();self.projector_k.eval()
        return self
    def forward(self, a,b=None):
        p=torch.relu(self.encoder_q(a))
        log=self.classifier(p)
        if b is None or not self.training: return log
        q=F.normalize(self.projector_q(p).float(),dim=1)
        with torch.no_grad():
            k=F.normalize(self.projector_k(torch.relu(self.encoder_k(b))).float(),dim=1)
        return log,q,k
    @torch.no_grad()
    def get_valid_queue(self):
        n=int(self.queue_count.item())
        return self.queue_features[:,:n].detach().clone(), self.queue_labels[:,:n].detach().clone()
    @torch.no_grad()
    def dequeue_and_enqueue(self,keys,labels):
        keys=F.normalize(keys.float(),dim=1)
        length=keys.shape[0]
        for ix in range(length):
            ptr=int(self.queue_ptr.item())
            self.queue_features[:,ptr]=keys[ix]
            self.queue_labels[:,ptr]=labels[ix]
            self.queue_ptr[0]=(ptr+1)%self.queue_size
            self.queue_count[0]=min(self.queue_size,int(self.queue_count.item())+1)
    @torch.no_grad()
    def update_momentum_encoder(self):
        self.ema_calls+=1
        for q,k in ((self.encoder_q,self.encoder_k),(self.projector_q,self.projector_k)):
            for p_q,p_k in zip(q.parameters(),k.parameters()):
                p_k.lerp_(p_q,1-self.momentum)


def run(args):
    root=Path(args.project_root).resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from src.engine.trainer_supcon import train_supcon_epoch,validate_supcon
    from src.losses.contrastive_loss import MultiLabelQueueContrastiveLoss
    from src.losses.losses import BCELoss
    from train_supcon import checkpoint_payload,restore_checkpoint,atomic_save
    rows=[]
    output=root/'outputs'/'csv'/'stage2'/'audits'
    output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    labels=torch.zeros(9,14)
    labels[0,0]=labels[1,0]=labels[2,1]=labels[3,1]=1
    labels[4,2]=labels[5,2]=labels[6,3]=labels[7,3]=1
    images=torch.randn(9,3,16,16)
    loader=DataLoader(TensorDataset(images,images+torch.randn_like(images)*0.01,labels),batch_size=2,shuffle=False)
    model=TinyMomentumModel()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-3)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    cls=BCELoss()
    con=MultiLabelQueueContrastiveLoss(temperature=.07)
    tr=train_supcon_epoch(model,loader,cls,con,optimizer,scaler,torch.device('cpu'),
                          beta=0.1,accumulation_steps=2,amp=False)
    check(tr.optimizer_steps==3 and model.ema_calls==3, 'EMA once per optimizer step (5 batches, accumulation 2)',rows)
    check(tr.seen_images==9 and tr.valid_queue_at_end==7, 'FIFO queue and all 9 images processed',rows)
    check(all(torch.isfinite(torch.tensor([tr.loss,tr.classification_loss,tr.contrastive_loss]))),
          'total / classification / contrastive finite',rows)
    check(all(p.grad is None for p in model.encoder_k.parameters()),'key encoder never receives gradients',rows)
    val_loader=DataLoader(TensorDataset(images,labels),batch_size=3)
    va=validate_supcon(model,val_loader,cls,torch.device('cpu'),amp=False)
    check(va['n_images']==9 and va['n_valid_classes']==4,'validation valid class mask correct',rows)
    check(all(0<=va[k]<=1 for k in ('mAP','macro_AUC','macro_F1','macro_Precision','macro_Recall')),
          'five validation metrics bounded',rows)
    check(len(va['thresholds'])==14 and len(va['per_class_AP'])==14,'14 per-class thresholds and AP',rows)
    check(model.training is False,'validation switches model to eval (key branch remains eval)',rows)
    from types import SimpleNamespace
    cfg=SimpleNamespace(dataset='VinDr-CXR',seed=42,loss='bce',wbce_mode=1,gamma=2.,alpha=None,
       gamma_pos=0.,gamma_neg=4.,margin=.05,image_size=16,proj_dim=16,queue_size=7,
       momentum=.9,temperature=.07,beta=.1)
    payload=checkpoint_payload(1,model,optimizer,scaler,va['mAP'],cfg,va)
    ckpt=output/'_task04a_smoke_temp.pth'
    try:
        atomic_save(payload,ckpt)
        newer=TinyMomentumModel()
        newer_opt=torch.optim.AdamW((p for p in newer.parameters() if p.requires_grad),lr=1e-3)
        newer_scaler=torch.amp.GradScaler('cuda',enabled=False)
        start,best=restore_checkpoint(ckpt,newer,newer_opt,newer_scaler,torch.device('cpu'),cfg)
        check(start==2 and abs(best-va['mAP'])<1e-8,'checkpoint save + resume preserves epoch / best',rows)
        check(int(newer.queue_count.item())==7 and torch.equal(newer.queue_features,model.queue_features),
              'resume preserves queue tensors and count',rows)
    finally:
        ckpt.unlink(missing_ok=True)

    if args.real_model:
        from src.models.builder import build_model
        device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        actual=build_model('lcl_densenet201',num_classes=14,pretrained=False,proj_dim=128,
                           queue_size=16,momentum=.999).to(device)
        actual_opt=torch.optim.AdamW((p for p in actual.parameters() if p.requires_grad),lr=1e-4)
        actual_scale=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
        v=torch.randn(2,3,args.image_size,args.image_size)
        lab=torch.zeros(2,14);lab[:,0]=1
        actual_loader=DataLoader(TensorDataset(v,v.clone(),lab),batch_size=2)
        result=train_supcon_epoch(actual,actual_loader,cls.to(device),con.to(device),actual_opt,
             actual_scale,device,beta=.1,accumulation_steps=1,amp=device.type=='cuda')
        check(result.optimizer_steps==1 and result.valid_queue_at_end==2 and result.loss>=0,
              f'real DenseNet201 one optimization step ({device})',rows)
        del actual,actual_opt
        if device.type=='cuda':torch.cuda.empty_cache()

    report={'status':'PASS','checks':len(rows),'dataset_test':'synthetic',
            'real_model_tested':args.real_model,'train':tr.__dict__,
            'validation':{k:v for k,v in va.items() if k not in ('per_class_AP','thresholds')}}
    (output/'task04a_checks.csv').write_text('check,status\n'+'\n'.join(f'{x["check"]},{x["status"]}' for x in rows),encoding='utf-8')
    (output/'task04a_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('TASK04A STATUS: PASS |',output)


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--project-root',default=str(Path(__file__).resolve().parents[1]))
    a.add_argument('--real-model',action='store_true')
    a.add_argument('--image-size',type=int,default=64)
    a.add_argument('--cpu',action='store_true')
    run(a.parse_args())
