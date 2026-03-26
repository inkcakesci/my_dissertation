# -*- coding: utf-8 -*-
"""
Fixed-parameter training entrypoint for LightPred.

Run:
    python -m src.lightpred.train_lightpred_default
"""

from __future__ import annotations

from argparse import Namespace
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lightpred.data import DATA_ROOT
from src.lightpred.train_lightpred import run_training


def build_default_args() -> Namespace:
    return Namespace(
        # training
        model_type="lightpred",
        model_name="lightpred",
        epochs=150,              # upper bound only; early stopping should stop earlier
        batch_size=8,            # seq_len increased -> reduce batch
        lr=1e-4,                 # more conservative for stability
        weight_decay=1e-3,
        seq_len=4096,            # longer context to cover up to 40d periods better
        include_time=False,
        jitter_std=0.01,         # small noise augmentation (assuming normalized flux)
        min_period=0.5,
        max_period=40.0,
        val_fraction=0.1,
        max_items=None,
        num_workers=0,
        seed=42,
        out_dir=str(DATA_ROOT / "models"),
        no_cuda=False,
        early_stop_patience=12,  # stop after 12 non-improving epochs
        early_stop_min_delta=1e-3,
        early_stop_min_epochs=20,
        monitor_metric="val_loss",
        loss_type="nll",

        # model params (reduced capacity for N~909)
        lstm_hidden=64,
        lstm_layers=1,
        lstm_bidirectional=False,
        d_model=64,
        n_heads=4,               # 64 divisible by 4
        tf_layers=2,
        ff_dim=128,
        dropout=0.2,
        cnn_channels=32,
        cnn_kernel_size=9,
    )

def main():
    args = build_default_args()
    run_training(args)


if __name__ == "__main__":
    main()
