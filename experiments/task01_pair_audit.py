from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = {
    'VinDr-CXR': [
        'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
        'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
        'Other lesion', 'Pleural effusion', 'Pleural thickening',
        'Pneumothorax', 'Pulmonary fibrosis',
    ],
    'VinDr-PCXR': [
        'Bronchitis', 'Brocho-pneumonia', 'Other disease', 'Bronchiolitis',
        'Situs inversus', 'Pneumonia', 'Pleuro-pneumonia',
        'Diagphramatic hernia', 'Tuberculosis', 'Congenital emphysema',
        'CPAM', 'Hyaline membrane disease', 'Mediastinal tumor', 'Lung tumor',
    ],
}
DEFAULT_SEEDS = (7, 42, 132, 2024, 3407)
SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp')


def args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2],
                   help='Project root (folder containing data/ and outputs/)')
    p.add_argument('--data-root', type=Path, default=None,
                   help='Folder containing VinDr-CXR/ and VinDr-PCXR/. Default: data/processed/ then data/')
    p.add_argument('--batch-size', type=int, default=8,
                   help='Micro-batch actually sent through the network; NOT accumulation effective batch')
    p.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    p.add_argument('--queue-size', type=int, default=0,
                   help='Optional FIFO label-queue audit; 0 disables (no embeddings and no GPU)')
    p.add_argument('--check-images', action='store_true', help='Check image stems in train/ and test/ (slower)')
    p.add_argument('--output-dir', type=Path, default=None,
                   help='Task01 output directory; default outputs/csv/stage2/audits')
    return p.parse_args()


def find_data_root(project_root: Path, specified: Path | None) -> Path:
    choices = [specified] if specified else [project_root / 'data' / 'processed', project_root / 'data']
    for item in choices:
        if item is not None and all((item / ds / 'labels').is_dir() for ds in CLASSES):
            return item.resolve()
    raise FileNotFoundError('Cannot find both dataset labels directories under: ' + ', '.join(str(x) for x in choices))


def load_split(csv_file: Path, labels: list[str], dataset: str, split: str):
    if not csv_file.is_file():
        raise FileNotFoundError(f'{dataset}/{split}: missing {csv_file}')
    df = pd.read_csv(csv_file, dtype={'image_id': 'string'}, encoding='utf-8-sig')
    missing = [x for x in ['image_id', *labels] if x not in df.columns]
    if missing:
        raise ValueError(f'{dataset}/{split}: columns missing {missing}')
    if df.empty:
        raise ValueError(f'{dataset}/{split}: empty CSV')
    ids = df['image_id']
    if ids.isna().any() or ids.str.strip().eq('').any():
        raise ValueError(f'{dataset}/{split}: null/empty image IDs')
    if ids.duplicated().any():
        raise ValueError(f'{dataset}/{split}: duplicate IDs: {ids[ids.duplicated()].head().tolist()}')
    numeric = df.loc[:, labels].apply(pd.to_numeric, errors='raise')
    values = numeric.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError(f'{dataset}/{split}: labels must be finite binary 0/1')
    return df, values.astype(np.uint8, copy=False)


def split_audit(ds: str, tables: dict, arrays: dict):
    sizes = []
    for split in ('train', 'val', 'test'):
        y = arrays[split]
        pos_per_img = y.sum(axis=1)
        sizes.append({
            'dataset': ds, 'split': split, 'n_images': len(y), 'n_classes': y.shape[1],
            'n_all_zero': int((pos_per_img == 0).sum()),
            'pct_all_zero': 100 * (pos_per_img == 0).mean(),
            'n_single_positive': int((pos_per_img == 1).sum()),
            'n_multi_positive': int((pos_per_img > 1).sum()),
            'total_positive_assignments': int(y.sum()),
            'mean_labels_per_image': float(pos_per_img.mean()),
            'valid_AP_labels': int((y.sum(0) > 0).sum()),
            'valid_AUC_labels': int(((y.sum(0) > 0) & (y.sum(0) < len(y))).sum()),
        })
    overlaps = []
    for a, b in itertools.combinations(('train', 'val', 'test'), 2):
        ids_a = set(tables[a]['image_id'].tolist())
        ids_b = set(tables[b]['image_id'].tolist())
        common = sorted(ids_a & ids_b)
        overlaps.append({'dataset': ds, 'split_a': a, 'split_b': b,
                         'n_overlapping_ids': len(common), 'example_ids': ';'.join(common[:5])})
    return sizes, overlaps


def frequency_audit(ds: str, arrays: dict, classes: list[str]):
    train_y = arrays['train']
    pos = train_y.sum(axis=0).astype(int)
    # Stable sort for ties; top 5, next 5, remaining 4, same as Stage1 EDA.
    rank_order = np.argsort(-pos, kind='stable')
    rank = np.empty(len(classes), dtype=int)
    rank[rank_order] = np.arange(1, len(classes) + 1)
    rows = []
    for j, label in enumerate(classes):
        r = int(rank[j])
        group = 'Head' if r <= 5 else 'Medium' if r <= 10 else 'Tail'
        x = {'dataset': ds, 'label': label, 'train_pos': int(pos[j]),
             'train_neg': int(len(train_y) - pos[j]),
             'train_prevalence_pct': 100 * pos[j] / len(train_y),
             'frequency_rank': r, 'frequency_group': group}
        for split in ('val', 'test'):
            p = int(arrays[split][:, j].sum())
            x[f'{split}_pos'] = p
            x[f'{split}_neg'] = int(len(arrays[split]) - p)
            x[f'{split}_AP_eligible'] = (p > 0)
            x[f'{split}_AUC_eligible'] = (0 < p < len(arrays[split]))
        rows.append(x)
    return rows


def image_stem_check(dataset_path: Path, tables: dict, ds: str):
    ret = []
    for split in ('train', 'val', 'test'):
        image_folder = dataset_path / ('test' if split == 'test' else 'train')
        if not image_folder.is_dir():
            ret.append({'dataset': ds, 'split': split, 'image_dir': str(image_folder),
                        'status': 'IMAGE_DIR_MISSING', 'n_missing': len(tables[split]), 'example': ''})
            continue
        stem_counts = Counter(p.stem for p in image_folder.iterdir()
                              if p.is_file() and p.suffix.lower() in SUFFIXES)
        stems = set(stem_counts)
        ids = tables[split]['image_id'].tolist()
        missing = [i for i in ids if i not in stems]
        duplicates = sum(count > 1 for count in stem_counts.values())
        status = 'PASS' if not missing and not duplicates else 'FAIL'
        ret.append({'dataset': ds, 'split': split, 'image_dir': str(image_folder),
                    'status': status, 'n_missing': len(missing),
                    'n_duplicate_stems': duplicates, 'example': ';'.join(missing[:5])})
    return ret


def simulate_pairs(ds: str, y: np.ndarray, classes: list[str], batch_size: int, seeds: list[int], queue_size: int):
    """Random-shuffle batch sampler; multi-seed deterministic and CPU-only.

    An anchor always has its own other-view positive (same image). It can have a
    *supervised cross-image* positive only if another image shares an active label.
    Queue is a FIFO of labels from PREVIOUS micro-batches, used only if enabled;
    this is an optimistic pair-availability simulation, not a model performance prediction.
    """
    n, c = y.shape
    summary, per_label, batches = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        q = np.zeros((0, c), dtype=np.uint8)
        counts = {'n_all_zero': 0, 'n_positive': 0, 'n_multi': 0,
                  'anchors_with_inbatch': 0, 'anchors_with_queue': 0,
                  'disease_anchors_with_inbatch': 0, 'disease_anchors_with_queue': 0,
                  'total_cross_image_inbatch_pairs_directed': 0,
                  'total_cross_image_queue_pairs': 0,
                  'batch_without_cross_image_positive': 0}
        label_seen = np.zeros(c, dtype=np.int64)
        label_inbatch = np.zeros(c, dtype=np.int64)
        label_queue = np.zeros(c, dtype=np.int64)
        for batch_ix, start in enumerate(range(0, n, batch_size)):
            index = order[start:start + batch_size]
            b = y[index]
            b32 = b.astype(np.int32)
            same_disease = (b32 @ b32.T) > 0
            np.fill_diagonal(same_disease, False)
            own_positive_counts = same_disease.sum(axis=1)
            has_other = own_positive_counts > 0
            disease = b.sum(1) > 0
            # Per label: for each positive anchor, does at least one OTHER image in this batch have that label?
            freq_batch = b.sum(axis=0).astype(int)
            label_seen += freq_batch
            label_inbatch += freq_batch * (freq_batch > 1)
            if len(q):
                queue_overlap = (b32 @ q.astype(np.int32).T) > 0
                queue_counts = queue_overlap.sum(axis=1)
                has_queue = queue_counts > 0
                freq_q = q.sum(axis=0).astype(int)
                label_queue += freq_batch * (freq_q > 0)
            else:
                queue_counts = np.zeros(len(b), dtype=int)
                has_queue = np.zeros(len(b), dtype=bool)
            counts['n_all_zero'] += int((~disease).sum())
            counts['n_positive'] += int(disease.sum())
            counts['n_multi'] += int((b.sum(1) > 1).sum())
            counts['anchors_with_inbatch'] += int(has_other.sum())
            counts['anchors_with_queue'] += int(has_queue.sum())
            counts['disease_anchors_with_inbatch'] += int((has_other & disease).sum())
            counts['disease_anchors_with_queue'] += int((has_queue & disease).sum())
            counts['total_cross_image_inbatch_pairs_directed'] += int(own_positive_counts.sum())
            counts['total_cross_image_queue_pairs'] += int(queue_counts.sum())
            counts['batch_without_cross_image_positive'] += int(not same_disease.any())
            batches.append({'dataset': ds, 'seed': seed, 'batch_index': batch_ix,
                            'n_images': len(b), 'all_zero_count': int((~disease).sum()),
                            'disease_count': int(disease.sum()),
                            'anchors_with_inbatch': int(has_other.sum()),
                            'anchors_with_queue': int(has_queue.sum()),
                            'cross_image_inbatch_pairs_directed': int(own_positive_counts.sum()),
                            'cross_image_queue_pairs': int(queue_counts.sum()),
                            'queue_fill_before_batch': len(q)})
            if queue_size:
                # Queue stores labels, not view embeddings; q only contains previous batches.
                q = np.concatenate([q, b], axis=0)[-queue_size:]
        nbatches = math.ceil(n / batch_size)
        s = {'dataset': ds, 'seed': seed, 'micro_batch_size': batch_size,
             'queue_size': queue_size, 'n_images': n, 'n_batches': nbatches,
             'n_disease_images': counts['n_positive'], 'n_all_zero_images': counts['n_all_zero'],
             'pct_all_zero': 100 * counts['n_all_zero'] / n,
             'pct_all_anchors_with_inbatch_positive': 100 * counts['anchors_with_inbatch'] / n,
             'pct_disease_anchors_with_inbatch_positive': 100 * counts['disease_anchors_with_inbatch'] / max(1, counts['n_positive']),
             'pct_all_anchors_with_queue_positive': 100 * counts['anchors_with_queue'] / n,
             'pct_disease_anchors_with_queue_positive': 100 * counts['disease_anchors_with_queue'] / max(1, counts['n_positive']),
             'pct_batches_without_cross_image_positive': 100 * counts['batch_without_cross_image_positive'] / nbatches,
             'cross_image_inbatch_pairs_directed': counts['total_cross_image_inbatch_pairs_directed'],
             'cross_image_queue_pairs': counts['total_cross_image_queue_pairs'],
             'same_image_view_pairs_guaranteed': n}
        summary.append(s)
        for j, label in enumerate(classes):
            pos = int(label_seen[j])
            per_label.append({'dataset': ds, 'seed': seed, 'label': label,
                              'n_positive_anchors': pos,
                              'anchors_with_same_label_inbatch': int(label_inbatch[j]),
                              'pct_same_label_inbatch': 100 * label_inbatch[j] / pos if pos else float('nan'),
                              'anchors_with_same_label_in_queue': int(label_queue[j]),
                              'pct_same_label_in_queue': 100 * label_queue[j] / pos if pos else float('nan')})
    return summary, per_label, batches


def safe_write(rows: list[dict], path: Path):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
    print(f'  [CSV] {path.name}: {len(rows)} rows')


def run(cfg):
    if cfg.batch_size < 1 or cfg.queue_size < 0 or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError('batch-size >=1, queue-size >=0, seeds must be unique')
    root = cfg.project_root.expanduser().resolve()
    data_root = find_data_root(root, cfg.data_root.expanduser().resolve() if cfg.data_root else None)
    output = (cfg.output_dir.expanduser().resolve() if cfg.output_dir
              else root / 'outputs' / 'csv' / 'stage2' / 'audits')
    output.mkdir(parents=True, exist_ok=True)
    print(f'Project : {root}\nData    : {data_root}\nOutput  : {output}')
    print(f'Batch   : {cfg.batch_size} | Queue audit: {cfg.queue_size} | Seeds: {cfg.seeds}')
    split_rows, overlap_rows, freq_rows, summary_rows, class_rows, batch_rows, image_rows = [], [], [], [], [], [], []
    status = 'PASS'
    checks = {}
    for ds, classes in CLASSES.items():
        print(f'\n===== {ds} =====')
        ds_path = data_root / ds
        csvs = {'train': ds_path / 'labels' / 'train_split.csv',
                'val': ds_path / 'labels' / 'val_split.csv',
                'test': ds_path / 'labels' / 'test_labels.csv'}
        tables, arrays = {}, {}
        for split, path in csvs.items():
            tables[split], arrays[split] = load_split(path, classes, ds, split)
            print(f'  {split:5s}: {len(tables[split]):6d} records ({path})')
        s, o = split_audit(ds, tables, arrays)
        split_rows.extend(s)
        overlap_rows.extend(o)
        freq_rows.extend(frequency_audit(ds, arrays, classes))
        if any(r['n_overlapping_ids'] for r in o):
            status = 'FAIL'
        if cfg.check_images:
            irows = image_stem_check(ds_path, tables, ds)
            image_rows.extend(irows)
            if any(r['status'] != 'PASS' for r in irows):
                status = 'FAIL'
        sm, cl, ba = simulate_pairs(ds, arrays['train'], classes, cfg.batch_size, cfg.seeds, cfg.queue_size)
        summary_rows.extend(sm)
        class_rows.extend(cl)
        batch_rows.extend(ba)
        cx = pd.DataFrame(sm)
        print('  All-zero train: {:.2f}% | disease anchors with same-label batch partner: {:.2f}% (5 seeds mean)'.format(
              100 * (arrays['train'].sum(1) == 0).mean(), cx['pct_disease_anchors_with_inbatch_positive'].mean()))
        print('  Test valid AP labels:', int((arrays['test'].sum(0)>0).sum()), '/14')
        checks[ds] = {'split_sizes': {s: len(tables[s]) for s in tables},
                      'all_zero_train': int((arrays['train'].sum(1)==0).sum()),
                      'split_overlap_pass': not any(r['n_overlapping_ids'] for r in o),
                      'valid_val_AP': int((arrays['val'].sum(0)>0).sum()),
                      'valid_test_AP': int((arrays['test'].sum(0)>0).sum())}
    safe_write(split_rows, output / 'dataset_splits.csv')
    safe_write(overlap_rows, output / 'split_overlap.csv')
    safe_write(freq_rows, output / 'train_label_frequency.csv')
    safe_write(summary_rows, output / 'batch_positive_pair_availability.csv')
    safe_write(class_rows, output / 'per_label_pair_availability.csv')
    safe_write(batch_rows, output / 'per_batch_pair_diagnostics.csv')
    if cfg.check_images:
        safe_write(image_rows, output / 'image_stem_check.csv')
    # Summarize rates as mean/std over shuffles; std is sample ddof=1.
    numeric = ['pct_all_anchors_with_inbatch_positive', 'pct_disease_anchors_with_inbatch_positive',
               'pct_all_anchors_with_queue_positive', 'pct_disease_anchors_with_queue_positive',
               'pct_batches_without_cross_image_positive']
    grouped = pd.DataFrame(summary_rows).groupby('dataset')[numeric].agg(['mean', 'std'])
    grouped.columns = [f'{name}_{stat}' for name, stat in grouped.columns]
    grouped.reset_index().to_csv(output / 'pair_availability_mean_std.csv', index=False, encoding='utf-8-sig')
    checks['metadata'] = {'created_utc': datetime.now(timezone.utc).isoformat(),
                          'project_root': str(root), 'data_root': str(data_root),
                          'batch_size': cfg.batch_size, 'queue_size': cfg.queue_size,
                          'seeds': cfg.seeds, 'check_images': cfg.check_images,
                          'status': status,
                          'interpretation': 'Same-image two-view positive is ALWAYS available. Cross-image pairs use shared positive disease labels ONLY; all-zero/all-zero does not count.',
                          'limitations': ['image_id disjoint does not prove patient/content independence',
                                          'FIFO queue audit simulates labels of previous batches, not actual learned features',
                                          'group Head/Medium/Tail uses 5/5/4 train frequency ranks, not fixed prevalence thresholds']}
    with (output / 'task01_report.json').open('w', encoding='utf-8') as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)
    # Human-facing summary.
    lines = [f'# Task 01 — Pair Audit — {status}', '',
             f'- Data root: `{data_root}`', f'- Micro-batch: **{cfg.batch_size}**; queue audit: **{cfg.queue_size}**',
             '- Positive pair definitions: **same-image two views** (guaranteed), and **different images sharing at least one positive disease label**.',
             '- Two all-zero images are **not** counted as supervised disease-positive pairs.',
             '- Seed shuffles: ' + ', '.join(map(str, cfg.seeds)), '',
             '## Dataset overview', '', '| Dataset | Train | Val | Test | All-zero train | Valid Test AP classes |',
             '|---|---:|---:|---:|---:|---:|']
    for ds in CLASSES:
        x = checks[ds]
        lines.append(f"| {ds} | {x['split_sizes']['train']} | {x['split_sizes']['val']} | {x['split_sizes']['test']} | {x['all_zero_train']} | {x['valid_test_AP']}/14 |")
    lines += ['', '## Important interpretations', '',
              '1. Same-image positive availability is 100% **by construction**, not evidence of cross-image supervision.',
              '2. Batch estimates are based on random permutations of the TRAIN label matrix, not GPU inference.',
              '3. Gradient accumulation does NOT add positives between individual micro-batches unless explicitly implemented.',
              '4. Overlap checks use image_id only; optionally check image stems with `--check-images`.',
              '5. Missing positive classes on validation/test must not be silently scored as AP=0.',
              '', '## PASS criteria', '',
              f'- Unique IDs and binary labels per split: PASS',
              f'- Split ID overlaps: {status if not cfg.check_images else "See split_overlap.csv"}',
              '- Image files: ' + ('checked; see image_stem_check.csv' if cfg.check_images else 'not checked (enable --check-images)'),
              '- Cross-image availability: descriptive metric; **zero availability for extreme rare labels is not a validation failure**.',
              '', '## Output files', '',
              '- dataset_splits.csv', '- split_overlap.csv', '- train_label_frequency.csv',
              '- batch_positive_pair_availability.csv', '- pair_availability_mean_std.csv',
              '- per_label_pair_availability.csv', '- per_batch_pair_diagnostics.csv',
              '- task01_report.json', '- TASK01_SUMMARY.md']
    (output / 'TASK01_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nTASK01 STATUS: {status} | See {output / "TASK01_SUMMARY.md"}')
    return status == 'PASS'


if __name__ == '__main__':
    try:
        ok = run(args_parser())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'\nTASK01 FAILED: {exc}', file=sys.stderr)
        sys.exit(2)
    sys.exit(0 if ok else 1)
