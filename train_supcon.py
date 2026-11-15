from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.data.dataset import CLASSES, VinDrDataset
from src.data.transforms import get_transforms
from src.data.contrastive_dataset import ContrastiveVinDrDataset
from src.engine.trainer_supcon import train_supcon_epoch, validate_supcon
from src.models.builder import build_model
from src.losses.losses import BCELoss, WBCELoss, FocalLossMultiLabel, AsymmetricLoss
from src.losses.contrastive_loss import MultiLabelQueueContrastiveLoss

METRICS = ('mAP','macro_AUC','macro_F1','macro_Precision','macro_Recall')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_cls_loss(args, train_labels):
    if args.loss == 'bce':
        return BCELoss()
    if args.loss == 'wbce':
        n = len(train_labels)
        pos = train_labels.sum(axis=0).astype(float)
        ratio = (n-pos)/np.maximum(pos,1.0)
        if args.wbce_mode == 1:
            w = np.sqrt(ratio)
        elif args.wbce_mode == 2:
            w = np.clip(ratio, None, 10.)
        elif args.wbce_mode == 3:
            w = np.clip(ratio, None, 50.)
        else:
            raise ValueError('W0 control omitted; use wbce-mode 1/2/3')
        return WBCELoss(torch.tensor(w,dtype=torch.float32))
    if args.loss == 'focal':
        return FocalLossMultiLabel(gamma=args.gamma,alpha=args.alpha)
    if args.loss == 'asl':
        return AsymmetricLoss(gamma_pos=args.gamma_pos,gamma_neg=args.gamma_neg,margin=args.margin)
    raise ValueError(args.loss)


def atomic_save(payload, path:Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def append_history(path, row):
    path.parent.mkdir(parents=True,exist_ok=True)
    write_header = not path.exists()
    with path.open('a',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def checkpoint_payload(epoch, model, optimizer, scaler, best_map, args, validation):
    return dict(
        epoch=epoch, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
        scaler_state=scaler.state_dict(), best_val_map=best_map, config=vars(args),
        validation=validation, rng_torch=torch.get_rng_state(),
        rng_numpy=np.random.get_state(), rng_python=random.getstate(),
        rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_checkpoint(path,model,optimizer,scaler,device,args):
    # Trusted local checkpoint only; optimizer and RNG states need full unpickling.
    try:
        state=torch.load(path,map_location='cpu',weights_only=False)
    except TypeError:
        state=torch.load(path,map_location='cpu')
    old=state['config']
    fixed=('dataset','seed','loss','wbce_mode','gamma','alpha','gamma_pos','gamma_neg','margin',
           'image_size','proj_dim','queue_size','momentum','temperature','beta')
    mismatches=[k for k in fixed if old.get(k)!=getattr(args,k)]
    if mismatches:
        raise ValueError(f'Cannot resume incompatible config fields: {mismatches}')
    model.load_state_dict(state['model_state'],strict=True)
    optimizer.load_state_dict(state['optimizer_state'])
    for st in optimizer.state.values():
        for k,v in st.items():
            if torch.is_tensor(v):
                st[k]=v.to(device)
    scaler.load_state_dict(state['scaler_state'])
    torch.set_rng_state(state['rng_torch'].cpu())
    np.random.set_state(state['rng_numpy'])
    random.setstate(state['rng_python'])
    if device.type=='cuda' and state.get('rng_cuda') is not None:
        torch.cuda.set_rng_state_all(state['rng_cuda'])
    return int(state['epoch'])+1,float(state['best_val_map'])


def run(args):
    if args.epochs < 1 or args.batch_size<1 or args.accumulation_steps<1 or args.beta<0:
        raise ValueError('Invalid epochs, batch size, accumulation steps or beta')
    if args.dataset not in CLASSES:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    set_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    amp=device.type=='cuda' and not args.no_amp
    project=Path(args.project_root).resolve()
    output_root=(Path(args.output_root).resolve() if args.output_root else project/'outputs')
    data_root=Path(args.data_root).resolve() if args.data_root else project/'data'/'processed'
    dataset_dir=data_root/args.dataset
    tag=args.run_name or (args.loss if args.loss!='wbce' else f'wbce_w{args.wbce_mode}')
    if not all(c.isalnum() or c in '_-' for c in tag):
        raise ValueError('run-name must contain only letters, digits, _ or -')
    save_dir=output_root/'checkpoints'/'stage2'/args.dataset/tag/f'seed{args.seed}'
    report_dir=output_root/'csv'/'stage2'/'training'/args.dataset/tag/f'seed{args.seed}'
    save_dir.mkdir(parents=True,exist_ok=True)
    report_dir.mkdir(parents=True,exist_ok=True)
    cfg_path=save_dir/'config.json'
    if cfg_path.exists() and not args.resume:
        raise FileExistsError(f'Run exists: {cfg_path}. Use --resume or a different --run-name.')
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(vars(args),indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Dataset: {args.dataset} | device={device} | AMP={amp}')
    if args.no_pretrained: print('WARNING: pretrained=False; for smoke tests only, not the final baseline comparison')
    if args.max_train_batches is not None or args.max_val_batches is not None:
        print('WARNING: PARTIAL DEBUG RUN; validation metrics are NOT comparable to full Stage 1.')
    print(f'Data: {dataset_dir}\nCheckpoint: {save_dir}\nCSV: {report_dir}')
    train_ds=ContrastiveVinDrDataset(dataset_dir,args.dataset,args.image_size,strict_images=True)
    val_ds=VinDrDataset(csv_file=dataset_dir/'labels'/'val_split.csv',
                        img_dir=dataset_dir/'train',dataset_name=args.dataset,
                        transform=get_transforms('val',args.image_size))
    def worker_seed(worker_id):
        seed=torch.initial_seed()%(2**32)
        random.seed(seed);np.random.seed(seed)
    generator=torch.Generator().manual_seed(args.seed)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,
                            drop_last=False,num_workers=args.num_workers,
                            pin_memory=device.type=='cuda',worker_init_fn=worker_seed,
                            generator=generator)
    val_loader=DataLoader(val_ds,batch_size=args.val_batch_size,shuffle=False,
                          num_workers=args.num_workers,pin_memory=device.type=='cuda')
    model=build_model('lcl_densenet201',num_classes=14,pretrained=not args.no_pretrained,
                      proj_dim=args.proj_dim,queue_size=args.queue_size,momentum=args.momentum).to(device)
    cls_loss=build_cls_loss(args,train_ds.labels.numpy()).to(device)
    con_loss=MultiLabelQueueContrastiveLoss(temperature=args.temperature).to(device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=args.lr,weight_decay=args.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    best_val_map=-float('inf')
    start_epoch=1
    if args.resume:
        path=save_dir/'latest.pth'
        if not path.is_file():
            raise FileNotFoundError(f'Resume requested but no checkpoint at {path}')
        start_epoch,best_val_map=restore_checkpoint(path,model,optimizer,scaler,device,args)
        print(f'Resumed from epoch {start_epoch-1}; previous best Val mAP={best_val_map:.5f}')
    history=report_dir/'history.csv'
    patience_count=0
    for epoch in range(start_epoch,args.epochs+1):
        epoch_start=time.perf_counter()
        print(f'\n=== {args.dataset} | EPOCH {epoch}/{args.epochs} | TRAIN ===',flush=True)
        # Epoch-specific shuffle supports exact epoch-boundary resume.
        generator.manual_seed(args.seed + epoch)
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        tr=train_supcon_epoch(model,train_loader,cls_loss,con_loss,optimizer,scaler,device,
                              beta=args.beta,accumulation_steps=args.accumulation_steps,
                              max_batches=args.max_train_batches,amp=amp,grad_clip=args.grad_clip,
                              epoch=epoch,total_epochs=args.epochs,progress_every=args.progress_every)
        print(f'=== {args.dataset} | EPOCH {epoch}/{args.epochs} | VALIDATION ===',flush=True)
        val=validate_supcon(model,val_loader,cls_loss,device,amp=amp,max_batches=args.max_val_batches,
                            epoch=epoch,total_epochs=args.epochs,progress_every=args.progress_every)
        peak=(torch.cuda.max_memory_allocated(device)/1024**3) if device.type=='cuda' else None
        row=dict(epoch=epoch,train_loss=tr.loss,train_cls_loss=tr.classification_loss,
                 train_con_loss=tr.contrastive_loss,val_loss=val['val_loss'],
                 n_valid_classes=val['n_valid_classes'],**{k:val[k] for k in METRICS},
                 optimizer_steps=tr.optimizer_steps,skipped_steps=tr.skipped_steps,
                 queue_count=tr.valid_queue_at_end,train_images_seen=tr.seen_images,
                 disease_anchors=tr.disease_anchors,crossimage_disease_anchors=tr.crossimage_disease_anchors,
                 peak_allocated_gib=peak)
        append_history(history,row)
        (report_dir/f'val_epoch{epoch:03d}.json').write_text(json.dumps(val,indent=2,allow_nan=True),encoding='utf-8')
        is_better=bool(np.isfinite(val['mAP']) and val['mAP']>best_val_map)
        if is_better:
            best_val_map=float(val['mAP'])
            patience_count=0
        else:
            patience_count+=1
        payload=checkpoint_payload(epoch,model,optimizer,scaler,best_val_map,args,val)
        atomic_save(payload,save_dir/'latest.pth')
        if is_better:
            atomic_save(payload,save_dir/'best.pth')
        print(f'Epoch {epoch}/{args.epochs}: loss={tr.loss:.4f} cls={tr.classification_loss:.4f} '
              f'con={tr.contrastive_loss:.4f} Val mAP={val["mAP"]:.4f} '
              f'valid={val["n_valid_classes"]}/14 queue={tr.valid_queue_at_end} '
              f'VRAM_peak={peak if peak is not None else "CPU"} best={is_better} '
              f'epoch_seconds={time.perf_counter()-epoch_start:.1f}',flush=True)
        if args.patience and patience_count>=args.patience:
            print(f'Early stopping: no improvement for {patience_count} epochs')
            break
    print(f'COMPLETE | best Val mAP={best_val_map:.5f} | {history}')


def arguments():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--project-root',default=str(ROOT))
    p.add_argument('--data-root',default=None,help='Directory with VinDr-CXR/ and VinDr-PCXR/')
    p.add_argument('--output-root',default=None,help='Persistent outputs folder, e.g. mounted Google Drive')
    p.add_argument('--dataset',choices=('VinDr-CXR','VinDr-PCXR'),required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--run-name',default=None)
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--val-batch-size',type=int,default=4)
    p.add_argument('--accumulation-steps',type=int,default=8)
    p.add_argument('--image-size',type=int,default=512)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--progress-every',type=int,default=10,help='Update live tqdm metrics every N batches')
    p.add_argument('--proj-dim',type=int,default=128)
    p.add_argument('--queue-size',type=int,default=1024)
    p.add_argument('--momentum',type=float,default=0.999)
    p.add_argument('--temperature',type=float,default=0.07)
    p.add_argument('--beta',type=float,default=0.1,help='Contrastive contribution')
    p.add_argument('--loss',choices=('bce','wbce','focal','asl'),default='bce')
    p.add_argument('--wbce-mode',type=int,default=1,choices=(1,2,3))
    p.add_argument('--gamma',type=float,default=2.0)
    p.add_argument('--alpha',type=float,default=None)
    p.add_argument('--gamma-pos',type=float,default=0.0)
    p.add_argument('--gamma-neg',type=float,default=4.0)
    p.add_argument('--margin',type=float,default=0.05)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--grad-clip',type=float,default=1.0)
    p.add_argument('--patience',type=int,default=0)
    p.add_argument('--max-train-batches',type=int,default=None,help='DEBUG ONLY: train first N batches')
    p.add_argument('--max-val-batches',type=int,default=None,help='DEBUG ONLY: val first N batches')
    p.add_argument('--no-pretrained',action='store_true',help='Debug only: do not download ImageNet weights')
    p.add_argument('--no-amp',action='store_true')
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--resume',action='store_true')
    return p.parse_args()

if __name__=='__main__':
    run(arguments())
