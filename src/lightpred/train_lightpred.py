# -*- coding: utf-8 -*-
"""
Train sequence models for Kepler lightcurves.

Example:
    python -m src.lightpred.train_lightpred --epochs 30 --batch-size 16
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from torch.utils.data import DataLoader, random_split

from src.lightpred.data import (
    build_index,
    collate_skip_none,
    LightPredDataset,
    DATA_ROOT,
)
from src.lightpred.model import (
    build_model_from_args,
    gaussian_nll,
    log_huber_loss,
    period_huber_loss,
    period_mae,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train sequence models for Prot regression.")
    p.add_argument("--model-type", type=str, default="lightpred", choices=["lightpred", "lstm", "cnn"])
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--include-time", action="store_true")
    p.add_argument("--jitter-std", type=float, default=0.0)
    p.add_argument("--min-period", type=float, default=0.5)
    p.add_argument("--max-period", type=float, default=40.0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=str(DATA_ROOT / "models"))
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument("--early-stop-min-epochs", type=int, default=0)
    p.add_argument("--monitor-metric", type=str, default="val_loss", choices=["val_loss", "val_mae"])
    p.add_argument(
        "--loss-type",
        type=str,
        default="nll",
        choices=["nll", "log_huber", "period_huber"],
    )

    # model params
    p.add_argument("--lstm-hidden", type=int, default=128)
    p.add_argument("--lstm-layers", type=int, default=2)
    p.add_argument("--lstm-bidirectional", action="store_true")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--tf-layers", type=int, default=2)
    p.add_argument("--ff-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--cnn-channels", type=int, default=32)
    p.add_argument("--cnn-kernel-size", type=int, default=9)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0

    for batch in loader:
        if batch is None:
            continue
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            mu, log_sigma = model(x)
            if model.loss_type == "nll":
                loss = gaussian_nll(mu, log_sigma, y)
            elif model.loss_type == "log_huber":
                loss = log_huber_loss(mu, y)
            elif model.loss_type == "period_huber":
                loss = period_huber_loss(mu, y)
            else:
                raise ValueError(f"Unsupported loss_type: {model.loss_type}")
            mae = period_mae(mu, y)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += loss.item()
        total_mae += mae.item()
        n_batches += 1

    if n_batches == 0:
        return float("inf"), float("inf")
    return total_loss / n_batches, total_mae / n_batches


def run_training(args):
    set_seed(args.seed)
    if args.model_name is None:
        args.model_name = str(args.model_type)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    items = build_index(
        min_period=args.min_period,
        max_period=args.max_period,
        max_items=args.max_items,
    )
    if len(items) < 10:
        raise RuntimeError("Not enough samples found. Please check your data directory.")
    print(f"[INFO] Dataset size: {len(items)}")

    dataset = LightPredDataset(
        items,
        seq_len=args.seq_len,
        include_time=args.include_time,
        jitter_std=args.jitter_std,
        seed=args.seed,
    )

    val_size = int(len(dataset) * args.val_fraction)
    if args.val_fraction > 0 and val_size == 0:
        val_size = 1
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("Train split is empty. Please reduce --val-fraction.")
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"[INFO] Train/val split: {len(train_ds)}/{len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_skip_none,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_skip_none,
        drop_last=False,
    )

    model = build_model_from_args(args).to(device)
    model.loss_type = args.loss_type

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.model_name
    best_path = out_dir / f"{prefix}_best.pt"
    best_loss_path = out_dir / f"{prefix}_best_loss.pt"
    best_mae_path = out_dir / f"{prefix}_best_mae.pt"
    last_path = out_dir / f"{prefix}_last.pt"

    best_val = float("inf")
    best_loss = float("inf")
    best_mae = float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = _run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_mae = _run_epoch(model, val_loader, optimizer, device, train=False)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_mae={train_mae:.4f} "
            f"val_loss={val_loss:.4f} val_mae={val_mae:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_mae": float(train_mae),
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
            }
        )

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "args": vars(args),
        }
        torch.save(payload, last_path)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(payload, best_loss_path)

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(payload, best_mae_path)

        monitored_value = val_loss if args.monitor_metric == "val_loss" else val_mae
        improved = monitored_value < (best_val - args.early_stop_min_delta)
        if improved:
            best_val = monitored_value
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(payload, best_path)
        else:
            epochs_without_improve += 1

        if (
            args.early_stop_patience > 0
            and epoch >= args.early_stop_min_epochs
            and epochs_without_improve >= args.early_stop_patience
        ):
            print(
                f"[INFO] Early stopping triggered at epoch {epoch:03d}. "
                f"Best {args.monitor_metric}={best_val:.4f} at epoch {best_epoch:03d}."
            )
            break

    log_path = out_dir / f"{prefix}_train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_mae", "val_loss", "val_mae"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    print(f"[INFO] Saved best checkpoint to: {best_path}")
    print(f"[INFO] Saved best-loss checkpoint to: {best_loss_path}")
    print(f"[INFO] Saved best-MAE checkpoint to: {best_mae_path}")
    if best_epoch > 0:
        print(f"[INFO] Best {args.monitor_metric} = {best_val:.4f} at epoch {best_epoch:03d}")
    print(f"[INFO] Saved training log to: {log_path}")


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
