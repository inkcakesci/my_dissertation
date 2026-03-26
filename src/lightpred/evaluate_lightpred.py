# -*- coding: utf-8 -*-
"""
Evaluate a trained sequence-model checkpoint and export prediction tables.

Example:
    python -m src.lightpred.evaluate_lightpred

Outputs by default:
    data/kepler/models/lightpred_predictions.csv
    data/kepler/models/lightpred_eval_summary.csv
    data/kepler/models/lightpred_baseline_compare_val.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lightpred.data import build_index, LightPredDataset, DATA_ROOT
from src.lightpred.model import build_model_from_args


BASELINE_CSV = DATA_ROOT / "baseline_periods_ls_acf.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained sequence-model checkpoint.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DATA_ROOT / "models" / "lightpred_best.pt"),
        help="Path to checkpoint (*.pt).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(DATA_ROOT / "models"),
        help="Directory to save evaluation CSV files.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--file-prefix", type=str, default=None)
    p.add_argument(
        "--baseline-csv",
        type=str,
        default=str(BASELINE_CSV),
        help="Optional baseline CSV for same-split comparison.",
    )
    return p.parse_args()


class EvalLightPredDataset(Dataset):
    """Wrap LightPredDataset and keep sample metadata for CSV export."""

    def __init__(
        self,
        items,
        *,
        seq_len: int,
        include_time: bool,
        seed: int,
    ) -> None:
        self.items = list(items)
        self.base = LightPredDataset(
            self.items,
            seq_len=seq_len,
            include_time=include_time,
            jitter_std=0.0,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        if sample is None:
            return None
        x, y = sample
        item = self.items[idx]
        return x, y, int(item.kic), float(item.period)


def collate_eval(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys, kics, periods = zip(*batch)
    return (
        torch.stack(xs, dim=0),
        torch.stack(ys, dim=0),
        torch.tensor(kics, dtype=torch.long),
        torch.tensor(periods, dtype=torch.float32),
    )


def _resolve_lengths(n_items: int, val_fraction: float) -> tuple[int, int]:
    val_size = int(n_items * val_fraction)
    if val_fraction > 0 and val_size == 0:
        val_size = 1
    train_size = n_items - val_size
    return train_size, val_size


def _split_items(items, *, val_fraction: float, seed: int):
    n_items = len(items)
    train_size, val_size = _resolve_lengths(n_items, val_fraction)
    if train_size <= 0:
        raise RuntimeError("Train split is empty. Please reduce val_fraction.")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_items, generator=generator).tolist()

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    return train_items, val_items


def _predict_split(model, loader, device: torch.device, split_name: str) -> pd.DataFrame:
    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, y, kics, true_period = batch
            x = x.to(device)
            mu, log_sigma = model(x)

            mu = mu.detach().cpu().numpy()
            log_sigma = np.clip(log_sigma.detach().cpu().numpy(), -6.0, 6.0)
            sigma = np.exp(log_sigma)
            y = y.detach().cpu().numpy()
            kics = kics.detach().cpu().numpy()
            true_period = true_period.detach().cpu().numpy()

            pred_period = np.exp(mu)
            pred_lo = np.exp(mu - sigma)
            pred_hi = np.exp(mu + sigma)

            for i in range(len(kics)):
                abs_err = abs(pred_period[i] - true_period[i])
                rel_err = abs_err / true_period[i] if true_period[i] > 0 else np.nan
                rows.append(
                    {
                        "split": split_name,
                        "kic": int(kics[i]),
                        "true_log_period": float(y[i]),
                        "true_period": float(true_period[i]),
                        "pred_log_period": float(mu[i]),
                        "pred_period": float(pred_period[i]),
                        "sigma_log_period": float(sigma[i]),
                        "pred_period_lo_1sigma": float(pred_lo[i]),
                        "pred_period_hi_1sigma": float(pred_hi[i]),
                        "abs_err": float(abs_err),
                        "rel_err": float(rel_err),
                    }
                )
    return pd.DataFrame(rows)


def _summarize_predictions(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "subset": label,
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MedianAbsErr": np.nan,
            "Bias": np.nan,
            "MedianBias": np.nan,
            "Frac(<10%)": np.nan,
            "Frac(<20%)": np.nan,
        }

    err = df["pred_period"] - df["true_period"]
    abs_err = err.abs()
    rel_err = abs_err / df["true_period"].replace(0, np.nan)
    return {
        "subset": label,
        "N": int(len(df)),
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MedianAbsErr": float(abs_err.median()),
        "Bias": float(err.mean()),
        "MedianBias": float(err.median()),
        "Frac(<10%)": float((rel_err < 0.10).mean()),
        "Frac(<20%)": float((rel_err < 0.20).mean()),
    }


def _summarize_baseline_on_split(pred_df: pd.DataFrame, baseline_csv: Path, model_label: str) -> pd.DataFrame:
    if pred_df.empty or not baseline_csv.exists():
        return pd.DataFrame()

    df_base = pd.read_csv(baseline_csv)
    if "kic" not in df_base.columns or "prot_label" not in df_base.columns:
        return pd.DataFrame()

    val_df = pred_df[pred_df["split"] == "val"].copy()
    if val_df.empty:
        return pd.DataFrame()

    merged = val_df.merge(df_base, on="kic", how="inner")
    if merged.empty:
        return pd.DataFrame()

    rows = []
    methods = [
        (model_label, "pred_period"),
        ("LS", "prot_ls"),
        ("ACF", "prot_acf"),
        ("GPS", "prot_gps"),
        ("QP-GP", "prot_qpgp"),
    ]
    for name, col in methods:
        if col not in merged.columns:
            continue
        tmp = merged[["prot_label", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if tmp.empty:
            continue
        err = tmp[col] - tmp["prot_label"]
        abs_err = err.abs()
        rel_err = abs_err / tmp["prot_label"].replace(0, np.nan)
        rows.append(
            {
                "method": name,
                "subset": "val",
                "N": int(len(tmp)),
                "MAE": float(abs_err.mean()),
                "RMSE": float(np.sqrt(np.mean(err**2))),
                "MedianAbsErr": float(abs_err.median()),
                "Bias": float(err.mean()),
                "MedianBias": float(err.median()),
                "Frac(<10%)": float((rel_err < 0.10).mean()),
                "Frac(<20%)": float((rel_err < 0.20).mean()),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    items = build_index(
        min_period=ckpt_args.get("min_period", 0.5),
        max_period=ckpt_args.get("max_period", 40.0),
        max_items=ckpt_args.get("max_items", None),
    )
    if len(items) == 0:
        raise RuntimeError("No samples available for evaluation.")

    train_items, val_items = _split_items(
        items,
        val_fraction=ckpt_args.get("val_fraction", 0.1),
        seed=ckpt_args.get("seed", 42),
    )

    class _Args:
        pass

    model_args = _Args()
    for key, value in ckpt_args.items():
        setattr(model_args, key, value)
    model = build_model_from_args(model_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    seq_len = ckpt_args.get("seq_len", 2048)
    include_time = ckpt_args.get("include_time", False)
    seed = ckpt_args.get("seed", 42)

    all_ds = EvalLightPredDataset(items, seq_len=seq_len, include_time=include_time, seed=seed)
    train_ds = EvalLightPredDataset(train_items, seq_len=seq_len, include_time=include_time, seed=seed)
    val_ds = EvalLightPredDataset(val_items, seq_len=seq_len, include_time=include_time, seed=seed)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_eval,
        drop_last=False,
    )

    pred_all = _predict_split(model, DataLoader(all_ds, **loader_kwargs), device, "all")
    pred_train = _predict_split(model, DataLoader(train_ds, **loader_kwargs), device, "train")
    pred_val = _predict_split(model, DataLoader(val_ds, **loader_kwargs), device, "val")

    pred_df = pd.concat([pred_all, pred_train, pred_val], ignore_index=True)

    summary_df = pd.DataFrame(
        [
            _summarize_predictions(pred_all, "all"),
            _summarize_predictions(pred_train, "train"),
            _summarize_predictions(pred_val, "val"),
        ]
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.file_prefix or ckpt_args.get("model_name") or ckpt_args.get("model_type", "lightpred")

    pred_path = out_dir / f"{prefix}_predictions.csv"
    summary_path = out_dir / f"{prefix}_eval_summary.csv"
    pred_df.to_csv(pred_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[INFO] Saved predictions to: {pred_path}")
    print(f"[INFO] Saved summary to: {summary_path}")
    print(summary_df.round(4).to_string(index=False))

    baseline_cmp = _summarize_baseline_on_split(pred_df, Path(args.baseline_csv), prefix)
    if not baseline_cmp.empty:
        baseline_cmp_path = out_dir / f"{prefix}_baseline_compare_val.csv"
        baseline_cmp.to_csv(baseline_cmp_path, index=False)
        print(f"[INFO] Saved baseline comparison to: {baseline_cmp_path}")
        print(baseline_cmp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
