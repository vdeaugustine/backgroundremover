"""Torch device selection for GUI inference."""

import torch

try:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        DEVICE_NAME = "NVIDIA GPU"
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        DEVICE_NAME = "Apple Silicon GPU"
    else:
        DEVICE = torch.device("cpu")
        DEVICE_NAME = "CPU"
except Exception:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"
