# -*- coding: utf-8 -*-
"""
Quick CUDA check for PyTorch environment.

Run:
    python -m src.lightpred.check_cuda
"""

from __future__ import annotations

import torch


def main():
    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_count: {torch.cuda.device_count()}")
        print(f"current_device: {torch.cuda.current_device()}")
        print(f"device_name: {torch.cuda.get_device_name(0)}")
        print(f"cuda_runtime_version: {torch.version.cuda}")


if __name__ == "__main__":
    main()
