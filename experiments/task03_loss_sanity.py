import argparse
import csv
import json
import sys
from pathlib import Path

import torch


def check(value, name, rows):
    passed = bool(value)
    rows.append(dict(check=name, status="PASS" if passed else "FAIL"))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if not passed:
        raise AssertionError(name)


def run(args):
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root))
    from src.losses.contrastive_loss import MultiLabelQueueContrastiveLoss

    out = root / "outputs" / "csv" / "stage2" / "audits"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    stats = {}
    try:
        torch.manual_seed(42)
        loss_fn = MultiLabelQueueContrastiveLoss(temperature=args.temperature)
        lab = torch.tensor([[1., 0., 0.], [1., 1., 0.], [0., 0., 1.], [0., 0., 0.], [0., 0., 0.]])
        mask = loss_fn.build_positive_weights(lab, lab, len(lab))
        check(torch.allclose(mask.diag(), torch.ones(5)), "self-keys are always positive", rows)
        check(mask[0, 1] > 0 and mask[1, 0] > 0, "shared disease creates supervised positive", rows)
        check(mask[0, 2] == 0 and mask[2, 0] == 0, "disjoint disease labels are not positive", rows)
        check(mask[3, 4] == 0 and mask[4, 3] == 0, "all-zero pairs are NOT cross-image positives", rows)
        check(mask[3, 0] == 0 and mask[0, 3] == 0, "all-zero versus disease is not positive", rows)
        try:
            loss_fn.build_positive_weights(lab, torch.ones(5, 2), 5)
            check(False, "wrong class dimension rejected", rows)
        except ValueError:
            check(True, "wrong class dimension rejected", rows)
        try:
            loss_fn.build_positive_weights(lab, torch.full((5, 3), 0.5), 5)
            check(False, "nonbinary labels rejected", rows)
        except ValueError:
            check(True, "nonbinary labels rejected", rows)

        b, d, c = 5, 16, 3
        q = torch.randn(b, d, requires_grad=True)
        k = torch.randn(b, d, requires_grad=True)
        l, s = loss_fn(q, k, lab, return_stats=True)
        l.backward()
        stats["empty_queue_loss"] = float(l.detach())
        check(torch.isfinite(l) and bool(q.grad.abs().sum() > 0), "empty queue finite and q gradients exist", rows)
        check(k.grad is None, "momentum keys detached from gradient", rows)
        check(s["queue_size_valid"] == 0 and s["n_allzero_anchors"] == 2, "empty queue stats correct", rows)
        q2 = torch.randn(b, d, requires_grad=True)
        queue_f = torch.randn(d, 7)
        queue_l = torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [1., 1., 0.], [0., 0., 0.], [0., 0., 0.]]).T
        l2, s2 = loss_fn(q2, k, lab, queue_f, queue_l, return_stats=True)
        l2.backward()
        stats["nonempty_queue_loss"] = float(l2.detach())
        check(torch.isfinite(l2) and s2["queue_size_valid"] == 7, "nonempty queue loss and stats correct", rows)
        check(q2.grad is not None and torch.isfinite(q2.grad).all(), "nonempty queue gradient finite", rows)
        rare_w = MultiLabelQueueContrastiveLoss(class_weights=torch.tensor([1., 2., 10.]))
        check(torch.isfinite(rare_w(q2.detach(), k.detach(), lab)), "optional class weights supported", rows)
        singleton = MultiLabelQueueContrastiveLoss()
        single_loss, single_stats = singleton(torch.randn(1, 16, requires_grad=True), torch.randn(1, 16), torch.zeros(1, 3), return_stats=True)
        check(torch.isfinite(single_loss) and single_stats["n_crossimage_positive_anchors"] == 0, "single all-zero image has only own key positive", rows)

        if args.check_real_images:
            from src.data.contrastive_dataset import ContrastiveVinDrDataset
            base = Path(args.data_root).resolve() if args.data_root else root / "data" / "processed"
            for dsname in ("VinDr-CXR", "VinDr-PCXR"):
                ds = ContrastiveVinDrDataset(base / dsname, dsname, image_size=args.image_size)
                view1, view2, labels = ds[0]
                check(view1.shape == (3, args.image_size, args.image_size) and view2.shape == view1.shape, f"{dsname} two view tensor shapes", rows)
                check(labels.shape == (14,) and torch.isfinite(view1).all() and torch.isfinite(view2).all(), f"{dsname} labels & views finite", rows)
                stats[f"{dsname}_train_images"] = len(ds)
        stats["checks_passed"] = len(rows)
        status = "PASS"
    except Exception as exc:
        status = "FAIL"
        stats["error"] = repr(exc)
        print("ERROR:", repr(exc))
    with (out / "task03_checks.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["check", "status"])
        w.writeheader(); w.writerows(rows)
    (out / "task03_report.json").write_text(json.dumps({"status": status, **stats}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TASK03 STATUS: {status} | {out}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--check-real-images", action="store_true")
    p.add_argument("--data-root", default=None)
    p.add_argument("--image-size", type=int, default=128)
    run(p.parse_args())
