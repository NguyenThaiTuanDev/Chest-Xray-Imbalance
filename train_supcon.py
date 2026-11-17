from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import hashlib
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


def loss_tag(args):
    """Stage 1-compatible loss identity; never select a loss automatically."""
    if args.loss_tag:
        tag=args.loss_tag
        if not all(c.isalnum() or c=='_' for c in tag):
            raise ValueError('loss-tag must contain only letters, digits, underscore')
        if not (tag == args.loss or tag.startswith(args.loss + '_')):
            raise ValueError('loss-tag must match the configured --loss family')
        return tag
    if args.loss == 'bce':
        return 'bce'
    if args.loss == 'wbce':
        return f'wbce_w{args.wbce_mode}'
    if args.loss == 'focal':
        # Label parameterizations explicitly when configured rather than
        # assuming they correspond to Stage 1 f0/f1/... preset definitions.
        return 'focal'
    if args.loss == 'asl':
        return 'asl'
    raise ValueError(args.loss)


def named_checkpoint(args):
    """E.g. best_bce_supcon_seed7.pth, best_wbce_w2_supcon_seed7.pth."""
    return f'best_{loss_tag(args)}_supcon_seed{args.seed}.pth'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path):
    # Checkpoints are trusted files from this project, not untrusted downloads.
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _validate_named_checkpoint(path: Path, history_path: Path, args):
    if not path.is_file():
        raise FileNotFoundError(path)
    ck = _load_checkpoint(path)
    h = read_checked_history(history_path, expected_epochs=args.epochs)
    cfg = ck.get('config', {})
    for key in ('dataset', 'seed', 'loss', 'image_size', 'beta', 'queue_size'):
        if cfg.get(key) != getattr(args, key):
            raise ValueError(f'Named checkpoint config mismatch: {key}')
    best_ep = int(ck['epoch'])
    expected_best_ep = int(h.loc[h['mAP'].astype(float).idxmax(), 'epoch'])
    if best_ep != expected_best_ep:
        raise ValueError(f'Named checkpoint is epoch {best_ep}, expected best epoch {expected_best_ep}')
    val_map = float(ck['validation']['mAP'])
    history_best = float(h['mAP'].max())
    if not np.isfinite(val_map) or abs(val_map-history_best) > 1e-8:
        raise ValueError('Named checkpoint Validation mAP differs from history')
    if abs(float(ck['best_val_map'])-history_best) > 1e-8:
        raise ValueError('Named checkpoint best_val_map differs from history')
    del ck
    return best_ep, history_best


def finalize_single_checkpoint(save_dir: Path, report_dir: Path, args):
    """After 15/15, keep exactly ONE .pth: best_<loss>_supcon_seedN.pth.

    Export and validate the named copy *before* removing best.pth/latest.pth.
    If anything fails, the temporary training checkpoints remain intact.
    """
    latest = save_dir/'latest.pth'
    best = save_dir/'best.pth'
    named = save_dir/named_checkpoint(args)
    hist = report_dir/'history.csv'

    if named.is_file():
        _validate_named_checkpoint(named, hist, args)
        if best.is_file() and _sha256(best) != _sha256(named):
            raise ValueError('Existing named checkpoint conflicts with best.pth')
    else:
        verify_run_files(save_dir, report_dir, args)
        if not best.is_file():
            raise FileNotFoundError(best)
        tmp = named.with_name(named.name + '.tmp')
        if tmp.exists():
            raise RuntimeError(f'Leftover temporary export; inspect first: {tmp}')
        with best.open('rb') as src, tmp.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if _sha256(tmp) != _sha256(best):
            raise RuntimeError('Export checksum mismatch; training files kept')
        _validate_named_checkpoint(tmp, hist, args)
        os.replace(tmp, named)
        _validate_named_checkpoint(named, hist, args)
        print(f'[EXPORTED] {named}', flush=True)

    # Existing completed named checkpoint is verified; temporary .pth can go.
    # Keep the JSON config, CSV history, validation JSON and other research outputs.
    for temporary in (latest, best):
        if temporary.is_file():
            temporary.unlink()
            print(f'[CLEANUP] removed temporary checkpoint: {temporary.name}', flush=True)
    pth_files = sorted(x.name for x in save_dir.glob('*.pth'))
    if pth_files != [named.name]:
        raise RuntimeError(f'Expected only {named.name}; found {pth_files}. Inspect manually.')
    print(f'[FINAL] one checkpoint only: {named}', flush=True)
    return named


def read_checked_history(path: Path, expected_epochs: int | None = None):
    import pandas as pd
    if not path.is_file():
        raise FileNotFoundError(f'Missing training history: {path}')
    h = pd.read_csv(path)
    if h.empty or not {'epoch', 'mAP'}.issubset(h.columns):
        raise ValueError(f'Invalid history: {path}')
    epochs = h['epoch'].astype(int).tolist()
    if epochs != list(range(1, len(epochs) + 1)):
        raise ValueError(f'History epochs not contiguous 1..n: {epochs}')
    if expected_epochs is not None and epochs[-1] != expected_epochs:
        raise ValueError(f'History ends at epoch {epochs[-1]}, expected {expected_epochs}')
    return h


def verify_run_files(save_dir: Path, report_dir: Path, args):
    """Verify epoch 15 and best mAP before exporting a stable named copy."""
    ck = _load_checkpoint(save_dir/'latest.pth')
    ep = int(ck['epoch'])
    h = read_checked_history(report_dir/'history.csv', expected_epochs=ep)
    if ep != args.epochs:
        raise ValueError(f'Run not complete: epoch={ep}/{args.epochs}')
    if not (save_dir/'best.pth').is_file():
        raise FileNotFoundError('best.pth missing')
    for key in ('dataset', 'seed', 'loss', 'image_size', 'beta', 'queue_size'):
        if ck['config'].get(key) != getattr(args, key):
            raise ValueError(f'Checkpoint config mismatch at {key}')
    if abs(float(h['mAP'].max()) - float(ck['best_val_map'])) > 1e-8:
        raise ValueError('Best Validation mAP mismatch between history and checkpoint')
    print(f'[VERIFIED] {args.dataset} seed{args.seed}: epochs=1..{ep}; best Val mAP={ck["best_val_map"]:.8f}', flush=True)
    del ck



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
    state=_load_checkpoint(path)
    old=state['config']
    fixed=('dataset','seed','loss','wbce_mode','gamma','alpha','gamma_pos','gamma_neg','margin',
           'image_size','proj_dim','queue_size','momentum','temperature','beta','loss_tag')
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
    if int(args.resume) + int(args.restart_unstarted) + int(args.finalize_only) > 1:
        raise ValueError('--resume, --restart-unstarted, --finalize-only are mutually exclusive')
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
    history_path=report_dir/'history.csv'
    latest_path=save_dir/'latest.pth'
    best_path=save_dir/'best.pth'
    named_path=save_dir/named_checkpoint(args)
    if named_path.is_file():
        # Already finished and compacted (or interrupted during cleanup).
        finalize_single_checkpoint(save_dir, report_dir, args)
        print(f'[SKIP] Verified completed run: {args.dataset} seed{args.seed}', flush=True)
        return
    if args.finalize_only:
        finalize_single_checkpoint(save_dir, report_dir, args)
        return
    if cfg_path.exists() and not args.resume:
        if args.restart_unstarted:
            if any(x.is_file() for x in (latest_path, best_path, history_path)):
                raise RuntimeError('Cannot restart: checkpoint/history already exists; inspect or use --resume')
            old_config=json.loads(cfg_path.read_text(encoding='utf-8'))
            fixed=('dataset','seed','loss','wbce_mode','gamma','alpha','gamma_pos',
                   'gamma_neg','margin','image_size','proj_dim','queue_size','momentum',
                   'temperature','beta','lr','weight_decay','batch_size','accumulation_steps','loss_tag')
            mismatches=[k for k in fixed if old_config.get(k)!=getattr(args,k)]
            if mismatches:
                raise RuntimeError(f'Orphan config not compatible with requested run: {mismatches}')
            print('[RECOVER] Explicitly restarting an unstarted run with same config.',flush=True)
        else:
            raise FileExistsError(f'Run exists: {cfg_path}. Use --resume, --restart-unstarted after inspection, or another --run-name.')
    elif args.resume and not latest_path.is_file():
        raise FileNotFoundError('Resume requested but latest.pth missing; never restart automatically.')
    elif not cfg_path.exists():
        if any(x.is_file() for x in (latest_path,best_path,history_path)):
            raise RuntimeError('Checkpoint/history exists without config.json; inspect before continuing.')
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
        path=latest_path
        start_epoch,best_val_map=restore_checkpoint(path,model,optimizer,scaler,device,args)
        h=read_checked_history(history_path,expected_epochs=start_epoch-1)
        if not best_path.is_file():
            raise FileNotFoundError('best.pth missing; refusing unsafe resume')
        if abs(float(h['mAP'].max())-best_val_map) > 1e-8:
            raise ValueError('History best mAP does not match latest checkpoint; refusing unsafe resume')
        print(f'Resumed from epoch {start_epoch-1}; previous best Val mAP={best_val_map:.5f}')
    history=history_path
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
    # A normal return may also occur with early-stopping enabled. Only produce
    # Stage 1-style named files after the full requested epoch count.
    if history.is_file() and latest_path.is_file():
        import pandas as pd
        last=int(pd.read_csv(history)['epoch'].iloc[-1])
        if last == args.epochs:
            finalize_single_checkpoint(save_dir,report_dir,args)
        else:
            print(f'[NOT EXPORTED] Last epoch {last}/{args.epochs}; keep best.pth/latest.pth for resume.',flush=True)
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
    p.add_argument('--loss-tag',default=None,help='Exact Stage 1 selected loss identifier, e.g. focal_f2 or asl_a1; does NOT set hyperparameters')
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
    p.add_argument('--restart-unstarted',action='store_true',help='Explicit recovery when only compatible config.json exists; never deletes any files')
    p.add_argument('--finalize-only',action='store_true',help='Convert an already completed best.pth/latest.pth run to one named checkpoint; do not train')
    p.add_argument('--resume',action='store_true')
    return p.parse_args()

if __name__=='__main__':
    run(arguments())
