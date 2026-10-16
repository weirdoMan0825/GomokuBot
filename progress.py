"""Progress-bar helper with a no-dependency fallback."""

from __future__ import annotations

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


def progress(iterable, **kwargs):
    """Wrap an iterable with tqdm when available."""
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, **kwargs)
