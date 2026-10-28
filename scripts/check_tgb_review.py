"""Manual check of tgbl-review (4.87M events, slow, several GB of RAM). NOT run by the tests.

    python scripts/check_tgb_review.py

Prints the observed counts so they can be compared by eye with Blueprint §6.2
(approx. 352K sources, 4.87M events, 1-d rating).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from gdb.data.tgb_loader import load_tgb  # noqa: E402


def main() -> None:
    t0 = time.time()
    d = load_tgb("tgbl-review")
    s = d.stream
    print(f"loaded in {time.time() - t0:.0f}s from {d.data_dir}")
    print(f"events: {len(s)}")
    print(f"unique sources: {len(np.unique(s.src))}, unique destinations: {len(np.unique(s.dst))}")
    print(f"edge_feat shape: {s.edge_feat.shape}, t range: {s.t.min():.0f}..{s.t.max():.0f}")
    print(
        f"train/val/test: {int(d.train_mask.sum())}/{int(d.val_mask.sum())}/{int(d.test_mask.sum())}"
    )


if __name__ == "__main__":
    main()
