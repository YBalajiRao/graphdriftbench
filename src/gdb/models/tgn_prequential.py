r"""Test-then-train TGN over a stream, returning the per-event prequential loss.

For every batch of ``batch_size`` events (in stream order): (1) the model scores the batch's
positives and one negative per event using only past events (memory is updated *after*
scoring); (2) the per-event loss $\ell_i=\mathrm{BCE}(\hat y_i^+,1)+\mathrm{BCE}(\hat y_i^-,0)$ is
recorded **before** any gradient step on that batch; (3) one optimiser step; (4) memory update.
Node information comes only from :class:`gdb.data.stream.StreamReader`: the initial table at
time 0, and an entity's features are installed only *after* the batch containing its first event
was scored. Negatives are drawn from destinations known so far. The learner receives no drift
labels, schedule or onset. Built from PyG's own ``TGNMemory`` (no torch_scatter / sparse).
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.nn import Linear
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader

from gdb.data.stream import ROLE_DST, EventStream, StreamReader
from gdb.models.tgn_smoke import GB, _Embedding, _Predictor, pick_device


def prequential_event_losses(
    stream: EventStream,
    *,
    batch_size: int = 200,
    memory_dim: int = 32,
    time_scale: float = 100.0,
    lr: float = 3e-3,
    seed: int = 0,
    use_node_feat: bool = True,
    min_free_gb: float = 2.5,
) -> dict:
    """Per-event prequential loss (length ``len(stream)``) plus device / memory / time info."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device, free_gb = pick_device(min_free_gb)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    reader = StreamReader(stream)
    table, role, initial = reader.initial_table()
    if role is None or initial is None:
        raise ValueError("stream needs node_role / node_initial (reserved id layout)")
    n_nodes = len(role)
    t_int = torch.from_numpy(np.rint(stream.t * time_scale).astype(np.int64)).to(device)
    msg_all = torch.from_numpy(stream.edge_feat).to(device)
    msg_dim = msg_all.size(-1)

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
    feat = proj = None
    if use_node_feat and table is not None:
        feat = torch.from_numpy(table).to(device)
        proj = Linear(feat.size(1), memory_dim).to(device)
    extra = list(proj.parameters()) if proj is not None else []
    params = {
        id(q): q for q in [*memory.parameters(), *gnn.parameters(), *pred.parameters(), *extra]
    }
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    nb = LastNeighborLoader(n_nodes, size=10, device=device)
    assoc = torch.empty(n_nodes, dtype=torch.long, device=device)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    dst_pool = torch.from_numpy(np.flatnonzero(initial & (role == ROLE_DST))).to(device)

    losses = np.empty(len(stream))
    pos0 = 0
    t0 = time.time()
    memory.train(), gnn.train(), pred.train()
    memory.reset_state()  # re-creates the message store on the target device
    nb.reset_state()
    for batch in reader.batches(batch_size):
        n = len(batch.src)
        src = torch.from_numpy(batch.src).to(device)
        dst = torch.from_numpy(batch.dst).to(device)
        t = t_int[pos0 : pos0 + n]
        msg = msg_all[pos0 : pos0 + n]
        neg = dst_pool[torch.randint(0, len(dst_pool), (n,), device=device)]
        opt.zero_grad()
        n_id = torch.cat([src, dst, neg]).unique()
        n_id, edge_index, e_id = nb(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        if proj is not None:
            z = z + proj(feat[n_id])
        z = gnn(z, last_update, edge_index, t_int[e_id], msg_all[e_id])
        pos = pred(z[assoc[src]], z[assoc[dst]])
        neg_out = pred(z[assoc[src]], z[assoc[neg]])
        per_event = bce(pos, torch.ones_like(pos)).squeeze(-1) + bce(
            neg_out, torch.zeros_like(neg_out)
        ).squeeze(-1)
        losses[pos0 : pos0 + n] = per_event.detach().cpu().numpy()  # recorded BEFORE the step
        memory.update_state(src, dst, t, msg)
        nb.insert(src, dst)
        per_event.mean().backward()
        opt.step()
        memory.detach()
        if len(batch.new_ids):  # features revealed only after the batch of the first event
            ids = torch.from_numpy(batch.new_ids).to(device)
            if feat is not None:
                feat[ids] = torch.from_numpy(batch.new_feat).to(device)
            new_dst = batch.new_ids[role[batch.new_ids] == ROLE_DST]
            if len(new_dst):
                dst_pool = torch.cat([dst_pool, torch.from_numpy(new_dst).to(device)])
        pos0 += n
    cuda = device.type == "cuda"
    return {
        "loss": losses,
        "device": device.type,
        "free_gb_before": free_gb,
        "peak_gb": torch.cuda.max_memory_allocated() / GB if cuda else 0.0,
        "seconds": time.time() - t0,
    }
