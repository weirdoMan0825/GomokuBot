"""Device selection helpers."""

from __future__ import annotations


def default_device() -> str:
    """Use CUDA by default when it is available."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device(device) -> str:
    """Return a readable device description."""
    import torch

    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        return f"{dev} ({torch.cuda.get_device_name(index)})"
    return str(dev)
