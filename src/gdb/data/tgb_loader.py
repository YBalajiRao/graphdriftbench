"""Load TGB link-prediction datasets from a configurable data root into an :class:`EventStream`.

TGB resolves its dataset folder as ``PROJ_DIR + root + name`` with ``PROJ_DIR`` fixed to the
installed package location. ``tgb_root`` re-points ``PROJ_DIR`` for the duration of a ``with``
block (the same trick as ``tgb_data/download_tgb.py``) and restores the original value on exit,
so the installed package is never modified and other code sees TGB's default afterwards.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gdb.data.stream import ROLE_DST, ROLE_SRC, EventStream

REPO_ROOT = Path(__file__).resolve().parents[3]


def default_data_root() -> Path:
    """Data root from ``configs/data/tgb.yaml`` (default ``D:\\research\\tgb_data``)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(REPO_ROOT / "configs" / "data" / "tgb.yaml")
    return Path(str(cfg.data_root))


@contextmanager
def tgb_root(root: str | Path | None = None) -> Iterator[Path]:
    """Temporarily point ``tgb.linkproppred.dataset.PROJ_DIR`` at ``root``; restore on exit."""
    import tgb.linkproppred.dataset as dataset_module

    root_path = Path(root if root is not None else default_data_root()).resolve()
    original = dataset_module.PROJ_DIR
    dataset_module.PROJ_DIR = str(root_path) + os.sep
    try:
        yield root_path
    finally:
        dataset_module.PROJ_DIR = original


@dataclass
class TGBData:
    """A TGB dataset as our stream plus the official chronological split masks."""

    name: str
    stream: EventStream
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    data_dir: str


def load_tgb(name: str, root: str | Path | None = None) -> TGBData:
    """Load ``name`` (e.g. ``"tgbl-wiki"``) from ``root`` (default: the configured data root).

    Returns an :class:`EventStream` (ids as in TGB: sources first, destinations after, float64
    times, float32 edge features) and boolean train/val/test masks over its events. The dataset
    must already be on disk under ``<root>/datasets/<name with _>``; nothing is downloaded into the
    repository.
    """
    from tgb.linkproppred.dataset import LinkPropPredDataset

    with tgb_root(root) as root_path:
        folder = root_path / "datasets" / name.replace("-", "_")
        if not folder.is_dir():  # TGB would otherwise prompt on stdin to download
            raise FileNotFoundError(
                f"{folder} not found; download TGB data first (nothing is fetched here)"
            )
        ds = LinkPropPredDataset(name=name, root="datasets" + os.sep, preprocess=True)
        data = ds.full_data
        src, dst = data["sources"], data["destinations"]
        n_ids = int(max(src.max(), dst.max())) + 1
        role = np.full(n_ids, -1, dtype=np.int8)
        role[np.unique(src)] = ROLE_SRC
        role[np.unique(dst)] = ROLE_DST
        stream = EventStream(
            src,
            dst,
            data["timestamps"].astype(np.float64),
            data["edge_feat"],
            node_ids=np.arange(n_ids, dtype=np.int64),
            node_role=role,
        )
        return TGBData(name, stream, ds.train_mask, ds.val_mask, ds.test_mask, str(ds.root))
