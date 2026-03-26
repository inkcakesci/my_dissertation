# -*- coding: utf-8 -*-
"""
Sequence models for Prot regression.

Available architectures:
- LightPred: LSTM branch + Transformer branch + fusion head
- LSTM-only: small recurrent baseline
- CNN-only: lightweight convolutional baseline

All models return log-period mean and a log-sigma tensor for a shared
training/evaluation interface. Point-estimate models emit zero log-sigma.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class LightPredModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_bidirectional: bool = False,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        seq_len: int = 2048,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * (2 if lstm_bidirectional else 1)

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        fusion_dim = lstm_out_dim + d_model
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq, input_dim)
        lstm_out, (hn, _) = self.lstm(x)
        if self.lstm.bidirectional:
            last_fwd = hn[-2]
            last_bwd = hn[-1]
            lstm_feat = torch.cat([last_fwd, last_bwd], dim=-1)
        else:
            lstm_feat = hn[-1]

        x_proj = self.input_proj(x)
        x_proj = self.pos_enc(x_proj)
        tf_out = self.transformer(x_proj)
        tf_feat = tf_out.mean(dim=1)

        fused = torch.cat([lstm_feat, tf_feat], dim=-1)
        out = self.head(fused)
        mu = out[:, 0]
        log_sigma = out[:, 1]
        return mu, log_sigma


class LSTMOnlyModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        lstm_bidirectional: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        out_dim = lstm_hidden * (2 if lstm_bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (hn, _) = self.lstm(x)
        if self.lstm.bidirectional:
            feat = torch.cat([hn[-2], hn[-1]], dim=-1)
        else:
            feat = hn[-1]
        mu = self.head(feat).squeeze(-1)
        log_sigma = torch.zeros_like(mu)
        return mu, log_sigma


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNOnlyModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        cnn_channels: int = 32,
        cnn_kernel_size: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        mid_channels = cnn_channels * 2
        top_channels = cnn_channels * 4
        self.features = nn.Sequential(
            ConvBlock(input_dim, cnn_channels, cnn_kernel_size, dropout),
            ConvBlock(cnn_channels, mid_channels, max(5, cnn_kernel_size - 2), dropout),
            ConvBlock(mid_channels, top_channels, max(3, cnn_kernel_size - 4), dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(top_channels, top_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(top_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        feat = self.features(x)
        mu = self.head(feat).squeeze(-1)
        log_sigma = torch.zeros_like(mu)
        return mu, log_sigma


def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_sigma = torch.clamp(log_sigma, min=-6.0, max=6.0)
    var = torch.exp(2.0 * log_sigma)
    nll = 0.5 * ((target - mu) ** 2 / var + 2.0 * log_sigma)
    return nll.mean()


def period_mae(mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_period = torch.exp(mu)
    true_period = torch.exp(target)
    return torch.mean(torch.abs(pred_period - true_period))


def log_huber_loss(mu: torch.Tensor, target: torch.Tensor, delta: float = 0.2) -> torch.Tensor:
    return F.smooth_l1_loss(mu, target, beta=delta)


def period_huber_loss(mu: torch.Tensor, target: torch.Tensor, delta: float = 2.0) -> torch.Tensor:
    pred_period = torch.exp(mu)
    true_period = torch.exp(target)
    return F.smooth_l1_loss(pred_period, true_period, beta=delta)


def build_model_from_args(args) -> nn.Module:
    model_type = getattr(args, "model_type", "lightpred")
    input_dim = 2 if getattr(args, "include_time", False) else 1
    if model_type == "lightpred":
        return LightPredModel(
            input_dim=input_dim,
            lstm_hidden=getattr(args, "lstm_hidden", 128),
            lstm_layers=getattr(args, "lstm_layers", 2),
            lstm_bidirectional=getattr(args, "lstm_bidirectional", False),
            d_model=getattr(args, "d_model", 128),
            n_heads=getattr(args, "n_heads", 4),
            num_layers=getattr(args, "tf_layers", 2),
            ff_dim=getattr(args, "ff_dim", 256),
            dropout=getattr(args, "dropout", 0.1),
            seq_len=getattr(args, "seq_len", 2048),
        )
    if model_type == "lstm":
        return LSTMOnlyModel(
            input_dim=input_dim,
            lstm_hidden=getattr(args, "lstm_hidden", 64),
            lstm_layers=getattr(args, "lstm_layers", 1),
            lstm_bidirectional=getattr(args, "lstm_bidirectional", False),
            dropout=getattr(args, "dropout", 0.1),
        )
    if model_type == "cnn":
        return CNNOnlyModel(
            input_dim=input_dim,
            cnn_channels=getattr(args, "cnn_channels", 32),
            cnn_kernel_size=getattr(args, "cnn_kernel_size", 9),
            dropout=getattr(args, "dropout", 0.1),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")
