"""Event-stream container shared by the synthetic generator, the TGB loader and the learners.

An :class:`EventStream` holds only what a learner may see: events, edge features and (optionally)
node features. Ground truth (drift schedule, block labels) lives elsewhere and is never attached.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROLE_SRC = 0
ROLE_DST = 1


@dataclass
class EventStream:
    """A continuous-time dynamic graph as time-sorted events $(u_i, v_i, t_i, \\mathbf{x}_i)$.

    Attributes:
        src, dst: global integer node ids (TGB convention: sources ``0..N_src-1``, destinations
            ``N_src..N_src+N_dst-1``; ids created later by node churn are appended after them).
        t: event times, non-decreasing, float64.
        edge_feat: ``(E, d_e)`` float32 edge features / marks.
        node_feat: optional ``(n_nodes, d_x)`` float32 node features, row ``i`` for ``node_ids[i]``.
        node_ids: ids that ``node_feat`` / ``node_role`` rows refer to.
        node_role: optional ``(n_nodes,)`` int8, 0 = source side, 1 = destination side.
        node_initial: optional ``(n_nodes,)`` bool. ``True`` = entity known at the start of the
            stream; ``False`` = reserved id whose entity (and features) may only be used after its
            first event (see :meth:`observed_node_table`). ``None`` (legacy layout) = all known.
    """

    src: np.ndarray
    dst: np.ndarray
    t: np.ndarray
    edge_feat: np.ndarray
    node_feat: np.ndarray | None = None
    node_ids: np.ndarray | None = None
    node_role: np.ndarray | None = None
    node_initial: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.src = np.ascontiguousarray(self.src, dtype=np.int64)
        self.dst = np.ascontiguousarray(self.dst, dtype=np.int64)
        self.t = np.ascontiguousarray(self.t, dtype=np.float64)
        self.edge_feat = np.ascontiguousarray(self.edge_feat, dtype=np.float32)
        n = len(self.src)
        if not (len(self.dst) == len(self.t) == len(self.edge_feat) == n):
            raise ValueError("src, dst, t and edge_feat must have the same length")
        if n > 1 and np.any(np.diff(self.t) < 0):
            raise ValueError("event times must be non-decreasing")

    def __len__(self) -> int:
        return len(self.src)

    @property
    def n_nodes(self) -> int:
        """Number of distinct node ids that appear in at least one event."""
        return int(len(np.unique(np.concatenate([self.src, self.dst]))))

    def window(self, start: int, stop: int) -> EventStream:
        """Events ``[start, stop)`` as a new stream (node tables are shared, not copied)."""
        return EventStream(
            self.src[start:stop],
            self.dst[start:stop],
            self.t[start:stop],
            self.edge_feat[start:stop],
            self.node_feat,
            self.node_ids,
            self.node_role,
            self.node_initial,
        )

    def observed_node_table(self, upto: int) -> tuple[np.ndarray, np.ndarray]:
        r"""Leak-free node table after processing ``events[:upto]``.

        Returns ``(feat, known)``: ``known[i]`` is True when row ``i`` is an initial entity or has
        appeared in ``events[:upto]``; features of the other rows are zeroed, so a learner cannot
        see future entities or their features. Rows are indexed by node id.
        """
        if self.node_feat is None:
            raise ValueError("stream has no node features")
        n = len(self.node_feat)
        known = np.ones(n, dtype=bool) if self.node_initial is None else self.node_initial.copy()
        known[self.src[:upto]] = True
        known[self.dst[:upto]] = True
        return np.where(known[:, None], self.node_feat, 0.0).astype(np.float32), known

    def hash(self) -> str:
        """SHA-256 over events, features and node tables (reproducibility checks)."""
        h = hashlib.sha256()
        for arr in (
            self.src,
            self.dst,
            self.t,
            self.edge_feat,
            self.node_feat,
            self.node_ids,
            self.node_role,
            self.node_initial,
        ):
            if arr is not None:
                h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def to_temporal_data(self, time_scale: float | None = None):
        """Export to ``torch_geometric.data.TemporalData`` (``msg`` = edge features).

        ``t`` is float32 by default; with ``time_scale`` it is ``round(t * time_scale)`` as int64
        (``TGNMemory`` keeps integer ``last_update`` buffers, so TGN wants integer times).
        """
        import torch
        from torch_geometric.data import TemporalData

        t = (
            self.t.astype(np.float32)
            if time_scale is None
            else np.rint(self.t * time_scale).astype(np.int64)
        )
        return TemporalData(
            src=torch.from_numpy(self.src),
            dst=torch.from_numpy(self.dst),
            t=torch.from_numpy(t),
            msg=torch.from_numpy(self.edge_feat),
        )

    def save(self, path: str | Path) -> None:
        """Save to a ``.pt`` file (dict of tensors)."""
        import torch

        d = {k: torch.from_numpy(v) for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}
        torch.save(d, path)

    @classmethod
    def load(cls, path: str | Path) -> EventStream:
        import torch

        d = torch.load(path, weights_only=True)
        return cls(**{k: v.numpy() for k, v in d.items()})
