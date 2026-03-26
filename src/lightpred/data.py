# -*- coding: utf-8 -*-
"""
LightPred data utilities.

Dataset inputs:
- data/kepler/lightcurves/kic#########.npz
  (time, flux, flux_err)
- data/kepler/mcquillan2014_catalog.csv (kic, prot)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "kepler"
CATALOG_PATH = DATA_ROOT / "mcquillan2014_catalog.csv"
LC_DIR = DATA_ROOT / "lightcurves"


@dataclass(frozen=True)
class LightPredItem:
    kic: int
    period: float
    path: Path


def _parse_kic_from_name(name: str) -> Optional[int]:
    if not name.startswith("kic"):
        return None
    stem = name.replace("kic", "").replace(".npz", "")
    try:
        return int(stem)
    except ValueError:
        return None


def build_index(
    catalog_path: Path = CATALOG_PATH,
    lc_dir: Path = LC_DIR,
    min_period: float = 0.2,
    max_period: float = 50.0,
    max_items: Optional[int] = None,
) -> List[LightPredItem]:
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog file not found: {catalog_path}. "
            "Please run prepare_kepler_mcquillan.py first."
        )
    if not lc_dir.exists():
        raise FileNotFoundError(
            f"Lightcurve directory not found: {lc_dir}. "
            "Please run prepare_kepler_mcquillan.py to download lightcurves."
        )

    df = pd.read_csv(catalog_path)
    if "kic" not in df.columns or "prot" not in df.columns:
        raise ValueError(f"Catalog missing required columns: {catalog_path}")

    df["kic"] = df["kic"].astype(int)
    label_map = dict(zip(df["kic"].values, df["prot"].values))

    items: List[LightPredItem] = []
    for npz_path in sorted(lc_dir.glob("kic*.npz")):
        kic = _parse_kic_from_name(npz_path.name)
        if kic is None:
            continue
        if kic not in label_map:
            continue
        period = float(label_map[kic])
        if not np.isfinite(period):
            continue
        if not (min_period <= period <= max_period):
            continue
        items.append(LightPredItem(kic=kic, period=period, path=npz_path))

    if max_items is not None:
        items = items[:max_items]

    return items


def _robust_scale(flux: np.ndarray) -> np.ndarray:
    flux = flux - np.nanmedian(flux)
    scale = np.nanmedian(np.abs(flux))
    if np.isfinite(scale) and scale > 0:
        flux = flux / scale
    return flux


def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    t = data["time"].astype(float)
    flux = data["flux"].astype(float)
    mask = np.isfinite(t) & np.isfinite(flux)
    t = t[mask]
    flux = flux[mask]
    if t.size < 20:
        raise ValueError("Not enough valid points.")
    order = np.argsort(t)
    return t[order], flux[order]


def _resample_uniform(
    t: np.ndarray,
    flux: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    t0 = t[0]
    t1 = t[-1]
    span = t1 - t0
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Invalid time span for resampling.")
    t_norm = (t - t0) / span
    grid = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    flux_interp = np.interp(grid, t_norm, flux).astype(np.float32)
    return grid.astype(np.float32), flux_interp


class LightPredDataset(Dataset):
    def __init__(
        self,
        items: Iterable[LightPredItem],
        seq_len: int = 2048,
        include_time: bool = False,
        jitter_std: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.items = list(items)
        self.seq_len = int(seq_len)
        self.include_time = bool(include_time)
        self.jitter_std = float(jitter_std)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        try:
            t, flux = _load_npz(item.path)
            flux = _robust_scale(flux)
            t_grid, flux_grid = _resample_uniform(t, flux, self.seq_len)
            if self.jitter_std > 0:
                flux_grid = flux_grid + self.rng.normal(
                    0.0, self.jitter_std, size=flux_grid.shape
                ).astype(np.float32)

            if self.include_time:
                x = np.stack([flux_grid, t_grid], axis=-1)
            else:
                x = flux_grid[:, None]

            target = np.log(item.period).astype(np.float32)
            return torch.from_numpy(x), torch.tensor(target, dtype=torch.float32)
        except Exception:
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)
