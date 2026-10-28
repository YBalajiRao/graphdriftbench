"""Minimal TGN link-prediction loop used as an export/VRAM smoke test (Step 1).

Built only from PyG's own ``TGNMemory`` / ``LastNeighborLoader`` / ``TransformerConv``: no
``torch_scatter``/``torch_sparse``/``pyg_lib``. Not the Step 4 backbone; it only checks that a
generated stream trains a TGN for a few epochs inside a small memory budget.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor
from torch.nn import Linear
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader

from gdb.data.stream import EventStream

GB = 2**30


class _Embedding(torch.nn.Module):
    def __init__(self, in_ch: int, out_ch: int, msg_dim: int, time_enc: torch.nn.Module) -> None:
        super().__init__()
        self.time_enc = time_enc
        self.conv = TransformerConv(
            in_ch, out_ch // 2, heads=2, dropout=0.1, edge_dim=msg_dim + time_enc.out_channels
        )

    def forward(
        self, x: Tensor, last_update: Tensor, edge_index: Tensor, t: Tensor, msg: Tensor
    ) -> Tensor:
        rel_t = (last_update[edge_index[0]] - t).to(x.dtype)
        return self.conv(x, edge_index, torch.cat([self.time_enc(rel_t), msg], dim=-1))


class _Predictor(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin_src, self.lin_dst, self.lin_out = (
            Linear(dim, dim),
            Linear(dim, dim),
            Linear(dim, 1),
        )

    def forward(self, z_src: Tensor, z_dst: Tensor) -> Tensor:
        return self.lin_out((self.lin_src(z_src) + self.lin_dst(z_dst)).relu())


def pick_device(min_free_gb: float = 2.5) -> tuple[torch.device, float | None]:
    """CUDA if at least ``min_free_gb`` GB are free right now, else CPU. Never touches other processes."""
    if not torch.cuda.is_available():
        return torch.device("cpu"), None
    free, _ = torch.cuda.mem_get_info()
    free_gb = free / GB
    return (torch.device("cuda") if free_gb >= min_free_gb else torch.device("cpu")), free_gb


def run_tgn_smoke(
    stream: EventStream,
    *,
    epochs: int = 10,
    batch_size: int = 200,
    memory_dim: int = 32,
    time_scale: float = 100.0,
    seed: int = 0,
    lr: float = 1e-3,
    min_free_gb: float = 2.5,
    use_node_feat: bool = False,
) -> dict:
    """Train TGN for ``epochs`` on the first 85% of ``stream``, report loss, AP on the rest and VRAM.

    Returns a dict with ``device``, ``free_gb_before`` (None on CPU-only machines), ``peak_gb``
    (peak allocated VRAM, 0 on CPU), per-epoch ``losses`` and the final held-out ``ap``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device, free_gb = pick_device(min_free_gb)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    data = stream.to_temporal_data(time_scale=time_scale).to(device)
    n_nodes = int(max(stream.src.max(), stream.dst.max())) + 1
    split = int(0.85 * len(data))
    train_data, test_data = data[:split], data[split:]
    dst_pool = torch.from_numpy(np.unique(stream.dst[:split])).to(
        device
    )  # destinations seen in training
    msg_dim = data.msg.size(-1)
    node_feat = node_proj = None
    if use_node_feat:  # static per-node features added to the memory state before the GNN
        if stream.node_feat is None:
            raise ValueError("use_node_feat needs stream.node_feat")
        # smoke test only: the table after the whole stream (offline); Step 4 uses StreamReader
        node_feat = torch.from_numpy(stream.observed_node_table(len(stream))[0]).to(device)
        node_proj = Linear(node_feat.size(1), memory_dim).to(device)

    memory = TGNMemory(
        n_nodes,
        msg_dim,
        memory_dim,
        memory_dim,
        message_module=IdentityMessage(msg_dim, memory_dim, memory_dim),
        aggregator_module=LastAggregator(),
    ).to(device)
    gnn = _Embedding(memory_dim, memory_dim, msg_dim, memory.time_enc).to(device)
    pred = _Predictor(memory_dim).to(device)
    extra = list(node_proj.parameters()) if node_proj is not None else []
    params = {
        id(q): q for q in [*memory.parameters(), *gnn.parameters(), *pred.parameters(), *extra]
    }  # time_enc is shared
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    loader_nb = LastNeighborLoader(n_nodes, size=10, device=device)
    assoc = torch.empty(n_nodes, dtype=torch.long, device=device)
    bce = torch.nn.BCEWithLogitsLoss()

    def step(batch, train: bool) -> tuple[Tensor, Tensor, Tensor]:
        neg = dst_pool[torch.randint(0, len(dst_pool), (batch.src.size(0),), device=device)]
        n_id = torch.cat([batch.src, batch.dst, neg]).unique()
        n_id, edge_index, e_id = loader_nb(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        if node_proj is not None:
            z = z + node_proj(node_feat[n_id])
        z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])
        pos = pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = pred(z[assoc[batch.src]], z[assoc[neg]])
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        loader_nb.insert(batch.src, batch.dst)
        loss = bce(pos, torch.ones_like(pos)) + bce(neg_out, torch.zeros_like(neg_out))
        if train:
            loss.backward()
            opt.step()
            memory.detach()
        return loss, pos.detach(), neg_out.detach()

    t0 = time.time()
    losses = []
    for _ in range(epochs):
        memory.train(), gnn.train(), pred.train()
        if node_proj is not None:
            node_proj.train()
        memory.reset_state()
        loader_nb.reset_state()
        total = 0.0
        for batch in TemporalDataLoader(train_data, batch_size=batch_size):
            opt.zero_grad()
            loss, _, _ = step(batch, True)
            total += float(loss) * batch.num_events
        losses.append(total / train_data.num_events)

    memory.eval(), gnn.eval(), pred.eval()
    if node_proj is not None:
        node_proj.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in TemporalDataLoader(test_data, batch_size=batch_size):
            _, pos, neg = step(batch, False)
            ps += [pos.sigmoid().cpu(), neg.sigmoid().cpu()]
            ys += [torch.ones(len(pos)), torch.zeros(len(neg))]
    ap = float(average_precision_score(torch.cat(ys).numpy(), torch.cat(ps).numpy()))
    cuda = device.type == "cuda"
    peak = torch.cuda.max_memory_allocated() / GB if cuda else 0.0
    reserved = torch.cuda.max_memory_reserved() / GB if cuda else 0.0
    return {
        "device": device.type,
        "free_gb_before": free_gb,
        "peak_gb": peak,
        "peak_reserved_gb": reserved,
        "losses": losses,
        "ap": ap,
        "seconds": time.time() - t0,
        "n_events": len(stream),
    }
