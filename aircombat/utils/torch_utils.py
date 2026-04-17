from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == 'cpu':
        return torch.device('cpu')

    if device_str == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but torch.cuda.is_available() is False.')
        try:
            _ = torch.zeros(1, device='cuda')
            return torch.device('cuda')
        except Exception as e:
            raise RuntimeError(f'CUDA requested but unusable in current PyTorch build: {e}')

    if device_str == 'auto':
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device='cuda')
                return torch.device('cuda')
            except Exception as e:
                print(f'[WARN] CUDA detected but unusable, fallback to CPU: {e}')
        return torch.device('cpu')

    raise ValueError(f'Unknown device option: {device_str}')


def to_torch(x, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    t = torch.as_tensor(x, device=device)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t
