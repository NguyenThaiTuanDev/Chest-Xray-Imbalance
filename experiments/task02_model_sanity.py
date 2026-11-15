from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import math
import os
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--mode', choices=['quick', 'cuda', 'cpu-forward'], default='quick')
    p.add_argument('--batch-sizes', type=str, default='2,4,8', help='GPU batch-size candidates, run in order')
    p.add_argument('--image-size', type=int, default=128, help='Smoke test only; 512 is a separate full-resolution test')
    p.add_argument('--queue-size', type=int, default=1024)
    p.add_argument('--proj-dim', type=int, default=128)
    p.add_argument('--momentum', type=float, default=0.999)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dataset', choices=['VinDr-CXR', 'VinDr-PCXR'], default='VinDr-CXR')
    p.add_argument('--output-dir', type=Path, default=None)
    return p.parse_args()


def add(checks, name, passed, details='', severity='error'):
    checks.append({'check': name, 'status': 'PASS' if passed else 'FAIL',
                   'severity': severity, 'details': str(details)})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f': {details}' if details else ''))


def check_equal_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    if sa.keys() != sb.keys():
        return False, 'state_dict keys differ'
    for name in sa:
        if not torch.equal(sa[name], sb[name]):
            return False, f'different tensor: {name}'
    return True, f'{len(sa)} state tensors identical'


def test_queue(model, checks, torch):
    """Verify raw storage chronology, wrap-around and overwrite including B > K."""
    k, d, c = model.queue_size, model.proj_dim, model.num_classes
    model.reset_queue()
    expected = deque(maxlen=k)
    device = model.queue_features.device
    # K+3 validates overwrite after wrapping and B>K handling.
    steps = [3, max(1, k - 1), 4, k + 3]
    counter = 0
    for step, count in enumerate(steps, 1):
        records = []
        for i in range(count):
            v = counter + i + 1
            records.append(v)
            expected.append(v)
        counter += count
        keys = torch.zeros(count, d, device=device)
        keys[:, 0] = torch.tensor(records, dtype=torch.float32, device=device)
        keys[:, 1] = 1.0  # normalization yields unique [v,1] vectors
        labels = torch.zeros(count, c, device=device)
        labels[:, 0] = torch.tensor(records, dtype=torch.float32, device=device)
        old_ptr = int(model.queue_ptr.item())
        model.dequeue_and_enqueue(keys, labels)
        new_ptr = int(model.queue_ptr.item())
        n = int(model.queue_count.item())
        ptr_ok = new_ptr == (old_ptr + count) % k
        n_ok = n == min(k, sum(steps[:step]))
        fs, ls = model.get_valid_queue()
        valid_shape = tuple(fs.shape) == (d, n) and tuple(ls.shape) == (c, n)
        # When full, oldest is at queue_ptr. When not full, oldest at 0.
        order = list(range(n)) if n < k else [(new_ptr + j) % k for j in range(n)]
        actual = ls[0, order].detach().cpu().tolist()
        values_ok = actual == list(expected)
        normalized = torch.nn.functional.normalize(keys[-min(k,count):], dim=1)
        norms_ok = torch.allclose(fs.norm(dim=0), torch.ones(n, device=fs.device), atol=1e-5)
        add(checks, f'FIFO step{step} enqueue={count}',
            ptr_ok and n_ok and valid_shape and values_ok and norms_ok,
            f'ptr={new_ptr}, count={n}, order={values_ok}, norms={norms_ok}')
    copy_features, copy_labels = model.get_valid_queue()
    before = model.queue_features[0, 0].item()
    copy_features[0, 0] += 1000
    add(checks, 'get_valid_queue returns independent copy',
        math.isclose(model.queue_features[0, 0].item(), before),
        'queue features are protected from changes to returned tensors')
    model.reset_queue()
    add(checks, 'reset_queue', int(model.queue_ptr.item()) == 0 and
        int(model.queue_count.item()) == 0 and torch.count_nonzero(model.queue_features) == 0 and
        torch.count_nonzero(model.queue_labels) == 0, 'queue cleared')
    try:
        model.dequeue_and_enqueue(torch.zeros(2, d + 1, device=device), torch.zeros(2, c, device=device))
        rejected = False
    except ValueError:
        rejected = True
    add(checks, 'queue rejects wrong feature dimensions', rejected)
    try:
        x = torch.zeros(1, d, device=device); x[0,0] = float('nan')
        model.dequeue_and_enqueue(x, torch.zeros(1, c, device=device))
        rejected = False
    except ValueError:
        rejected = True
    add(checks, 'queue rejects NaN feature', rejected)


def test_momentum(model, checks, torch):
    eq, desc = check_equal_state(model.encoder_q, model.encoder_k)
    pq, descp = check_equal_state(model.projector_q, model.projector_k)
    add(checks, 'momentum initialization copies encoder/projector', eq and pq, desc + ', ' + descp)
    key_frozen = all(not x.requires_grad for x in model.encoder_k.parameters()) and \
                 all(not x.requires_grad for x in model.projector_k.parameters())
    add(checks, 'key encoder and key projector frozen', key_frozen)
    model.train()
    add(checks, 'key branches stay in eval mode after model.train()',
        model.training and not model.encoder_k.training and not model.projector_k.training)
    # Small tensor-based exact EMA check; no full model clone.
    q = next(model.projector_q.parameters())
    k = next(model.projector_k.parameters())
    q_original = q.detach().clone()
    k_original = k.detach().clone()
    with torch.no_grad():
        q.add_(1.0)
        wanted = k_original * model.momentum + q.detach() * (1 - model.momentum)
        model.update_momentum_encoder()
        ok = torch.allclose(k.detach(), wanted, atol=1e-6, rtol=1e-5)
        q.copy_(q_original)
        k.copy_(k_original)
    add(checks, 'EMA update formula k = m*k + (1-m)*q', ok)
    # Verify the key modules remain frozen after update.
    add(checks, 'EMA does not enable key gradients',
        all(not p.requires_grad for p in model.encoder_k.parameters()))


def test_forward(model, checks, torch, *, device, batch_size, image_size, fp16):
    from contextlib import nullcontext
    model.to(device)
    model.train()
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    xq = torch.randn(batch_size, 3, image_size, image_size, device=device)
    xk = torch.randn(batch_size, 3, image_size, image_size, device=device)
    labels = torch.zeros(batch_size, model.num_classes, device=device)
    labels[:, 0] = 1
    model.zero_grad(set_to_none=True)
    autocast = torch.autocast('cuda', dtype=torch.float16) if fp16 else nullcontext()
    with autocast:
        logits, q, k = model(xq, xk)
        # Sanity-only objective. This is NOT the research contrastive objective.
        cls = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels)
        align = (1.0 - (q.float() * k.float()).sum(dim=1)).mean()
        loss = cls + 0.1 * align
    shape_ok = (tuple(logits.shape) == (batch_size, model.num_classes) and
                tuple(q.shape) == (batch_size, model.proj_dim) and
                tuple(k.shape) == (batch_size, model.proj_dim))
    norm_ok = torch.allclose(q.float().norm(dim=1), torch.ones(batch_size, device=device), atol=1e-3) and \
              torch.allclose(k.float().norm(dim=1), torch.ones(batch_size, device=device), atol=1e-3)
    finite = bool(torch.isfinite(loss).item())
    if fp16:
        # GradScaler for CUDA half precision.
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        scaler.scale(loss).backward()
        scaler.unscale_(torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=1e-5))
    else:
        loss.backward()
    q_grad = any(p.grad is not None and bool(torch.isfinite(p.grad).all().item()) for p in model.encoder_q.parameters())
    k_no_grad = all(p.grad is None for p in model.encoder_k.parameters()) and \
                all(p.grad is None for p in model.projector_k.parameters())
    add(checks, f'{device.type} B{batch_size} forward shapes', shape_ok)
    add(checks, f'{device.type} B{batch_size} normalized embeddings', bool(norm_ok))
    add(checks, f'{device.type} B{batch_size} finite sanity loss', finite, f'loss={loss.item():.6f}')
    add(checks, f'{device.type} B{batch_size} gradients q-only', q_grad and k_no_grad)
    model.dequeue_and_enqueue(k.detach(), labels)
    add(checks, f'{device.type} B{batch_size} queue after forward',
        int(model.queue_count.item()) == batch_size)
    model.eval()
    with torch.inference_mode():
        with autocast if not fp16 else torch.autocast('cuda', dtype=torch.float16):
            predicted = model(xq)
    add(checks, f'{device.type} B{batch_size} inference logits shape',
        tuple(predicted.shape) == (batch_size, model.num_classes))
    if device.type == 'cuda':
        torch.cuda.synchronize()
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    else:
        max_alloc = None; max_reserved = None
    seconds = time.perf_counter() - t0
    return {'batch_size':batch_size, 'image_size':image_size, 'device':device.type,
            'precision':'fp16' if fp16 else 'fp32', 'status':'PASS', 'seconds':round(seconds,3),
            'peak_allocated_gib':max_alloc,'peak_reserved_gib':max_reserved,
            'message':'synthetic image smoke test (not training benchmark)'}


def save_csv(path, records, headers):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)


def main():
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    output = (args.output_dir or root/'outputs'/'csv'/'stage2'/'audits').resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (root/'src'/'models'/'builder.py').is_file():
        raise SystemExit('Cannot find src/models/builder.py under --project-root: ' + str(root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    global torch
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    from src.models.builder import MultiLabelLCL
    # Read literal class constants without importing cv2/albumentations from dataset.py.
    dataset_source = root/'src'/'data'/'dataset.py'
    tree = ast.parse(dataset_source.read_text(encoding='utf-8'))
    class_lists = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('VINDR_CXR_CLASSES','VINDR_PCXR_CLASSES'):
                    class_lists[target.id] = ast.literal_eval(node.value)
    const_name = 'VINDR_CXR_CLASSES' if args.dataset == 'VinDr-CXR' else 'VINDR_PCXR_CLASSES'
    if const_name not in class_lists:
        raise RuntimeError(f'Cannot read {const_name} from {dataset_source}')
    labels = class_lists[const_name]
    batch_sizes = [int(x) for x in args.batch_sizes.split(',') if x.strip()]
    if not batch_sizes or min(batch_sizes) < 1: raise SystemExit('Invalid --batch-sizes')
    if args.queue_size < 1 or args.image_size < 64: raise SystemExit('Require queue-size >=1 and image-size >=64')
    checks=[]; trials=[]
    print('PROJECT:',root)
    print('MODE:',args.mode,'| device CUDA:',torch.cuda.is_available())
    print('SOURCE: src/models/builder.py (read only)')
    print('OUTPUT:',output)
    print('DATASET:',args.dataset,'| classes:',len(labels))
    add(checks,'dataset has 14 labels',len(labels)==14)
    t0=time.perf_counter()
    try:
        model=MultiLabelLCL(num_classes=len(labels),proj_dim=args.proj_dim,
                            queue_size=args.queue_size,momentum=args.momentum,pretrained=False)
        add(checks,'DenseNet201 constructor',True,'pretrained=False; no download')
        dim=model.classifier.in_features
        add(checks,'DenseNet201 feature dim',dim==1920,f'dim={dim}')
        add(checks,'classifier output size',model.classifier.out_features==14)
        add(checks,'projector output size',model.projector_q[-1].out_features==args.proj_dim)
        num_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total=sum(p.numel() for p in model.parameters())
        print('PARAMETERS: trainable=',num_trainable, 'total=',num_total)
        test_momentum(model,checks,torch)
        test_queue(model,checks,torch)
        # Explicit CUDA request does not silently downgrade to CPU.
        if args.mode=='cuda' and not torch.cuda.is_available():
            add(checks,'CUDA available',False,'CUDA mode explicitly requested but no CUDA GPU detected')
        elif args.mode in ('cuda','cpu-forward'):
            dev = torch.device('cuda' if args.mode=='cuda' else 'cpu')
            selected = batch_sizes if args.mode=='cuda' else batch_sizes[:1]
            for b in selected:
                if args.mode=='cuda' and dev.type=='cuda':
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    print(f'CUDA: {torch.cuda.get_device_name(0)}; free {free_mem/2**30:.2f} GiB / total {total_mem/2**30:.2f} GiB')
                try:
                    model.reset_queue()
                    record=test_forward(model,checks,torch,device=dev,batch_size=b,
                                       image_size=args.image_size,fp16=dev.type=='cuda')
                    trials.append(record)
                    print('TRIAL:',record)
                except torch.cuda.OutOfMemoryError as e:
                    print('OOM for batch:',b)
                    trials.append({'batch_size':b,'image_size':args.image_size,'device':dev.type,
                                   'precision':'fp16' if dev.type=='cuda' else 'fp32',
                                   'status':'OOM','seconds':None, 'peak_allocated_gib':None,
                                   'peak_reserved_gib':None, 'message':str(e)[:400]})
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception as e:
                    trials.append({'batch_size':b,'image_size':args.image_size,'device':dev.type,
                                   'precision':'fp16' if dev.type=='cuda' else 'fp32',
                                   'status':'FAIL','seconds':None, 'peak_allocated_gib':None,
                                   'peak_reserved_gib':None, 'message':str(e)[:400]})
                    print(traceback.format_exc())
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            if args.mode=='cuda' and torch.cuda.is_available():
                add(checks,'at least one CUDA forward/backward batch passes',
                    any(r['status']=='PASS' for r in trials))
            elif args.mode=='cpu-forward':
                add(checks,'CPU forward/backward batch passes',
                    any(r['status']=='PASS' for r in trials))
    except Exception as e:
        add(checks,'model-level audit exception',False,traceback.format_exc())
    status='PASS' if all(x['status']=='PASS' for x in checks) else 'FAIL'
    result={'task':'Stage2 Task02 Model/Momentum/Queue Audit', 'status':status,
            'mode':args.mode,'dataset':args.dataset,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'project_root':str(root),'args':vars(args) | {'project_root':str(root),'output_dir':str(output)},
            'torch_version':torch.__version__, 'cuda_available':torch.cuda.is_available(),
            'seconds':round(time.perf_counter()-t0,2), 'checks':checks, 'trials':trials}
    suffix=f'task02_{args.mode}_{args.dataset}'
    save_csv(output/f'{suffix}_checks.csv',checks,['check','status','severity','details'])
    save_csv(output/f'{suffix}_vram.csv',trials,
             ['batch_size','image_size','device','precision','status','seconds','peak_allocated_gib','peak_reserved_gib','message'])
    with open(output/f'{suffix}_report.json','w',encoding='utf-8') as f: json.dump(result,f,indent=2,ensure_ascii=False)
    lines=[f'# Task 02 — {args.mode.upper()} — {args.dataset}',
           f'**Status:** {status}',f'**Source:** `{root / "src/models/builder.py"}`',
           f'**Checks:** {sum(x["status"]=="PASS" for x in checks)}/{len(checks)} PASS',
           '', '| Check | Result | Notes |','|---|---|---|']
    for x in checks:
        lines.append(f'| {x["check"]} | {x["status"]} | {x["details"].replace(chr(10)," ")[:120]} |')
    if trials:
        lines += ['', '## Forward/backward smoke trials', '',
                  '| Batch | Image size | Status | Peak allocated GiB | Peak reserved GiB |',
                  '|---:|---:|---|---:|---:|']
        for r in trials:
            lines.append(f'| {r["batch_size"]} | {r["image_size"]} | {r["status"]} | '
                         f'{r["peak_allocated_gib"] if r["peak_allocated_gib"] is not None else "—"} | '
                         f'{r["peak_reserved_gib"] if r["peak_reserved_gib"] is not None else "—"} |')
    lines += ['', '> Quick PASS checks construction, EMA and queue only. CUDA PASS requires a real GPU forward/backward.',
              '> Synthetic-image smoke test does not establish viability at image_size=512.',
              '> Task 02 does not use train/val/test labels and does not train a research model.']
    (output/f'{suffix}_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('TASK02 STATUS:',status,'|',output)
    return 0 if status=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
