r"""Continuous-time dynamic stochastic block model with exact ground truth (Blueprint §6.1).

Intensity of an event between nodes $u$ (source side, block $a$) and $v$ (destination side, block $b$):

$$\lambda_{uv}(t) = \lambda_0\,\frac{B_{ab}(t)\,\theta_u\theta_v\,e^{\beta_h(t)\cos(\mathbf x_u,\mathbf x_v)}}
{c_{ab}(t)\;Z(t)},\qquad
c_{ab} = \frac{\sum_{u\in a,v\in b}\theta_u\theta_v e^{\beta_h\cos(\mathbf x_u,\mathbf x_v)}}{\Theta_a\Theta_b},$$

with $\Theta_a=\sum_{u\in a}\theta_u$ and $Z=\sum_{ab}B_{ab}\Theta_a\Theta_b$. The correction $c_{ab}$ makes the
expected number of events per block pair exactly $\lambda_0 B_{ab}\Theta_a\Theta_b/Z$ per unit time, whatever
the homophily $\beta_h$ is, so $B$ is the block-level law and $\beta_h$ only redistributes events inside
a block pair. Edge marks are independent of the endpoints: $\mathbf x_e\sim\mathcal N(\mu_e,\sigma_e^2 I)$.

Simulation is exact block-pair Poisson thinning in $O(\#\text{events})$: for each block pair draw
$\mathrm{Poisson}(\lambda^{prop}_{ab}\Delta t)$ proposals, pick endpoints proportionally to $\theta$, and accept with
probability $e^{\beta_h\cos(\mathbf x_u,\mathbf x_v) - |\beta_h|}$ (the proposal rate is divided by the mean acceptance). The law
is piecewise constant between breakpoints (onsets, ramp steps, ends), so ramps are staircase approximations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from gdb.data.stream import ROLE_DST, ROLE_SRC, EventStream
from gdb.synth.injectors import (
    ALPHA,
    BETA,
    GAMMA,
    DriftSchedule,
    Regime,
    apply_event,
    beta_target,
    delta_permutation,
)


@dataclass
class DSBMConfig:
    """Generator parameters.

    ``mode="bipartite"``: sources ``0..n_src-1``, destinations ``n_src..n_src+n_dst-1`` (TGB
    convention), ``B`` has shape ``(k_src, k_dst)``. ``mode="unipartite"``: one node set of size
    ``n_src`` with ``k_src`` blocks, symmetric ``B``; ``n_dst``/``k_dst`` are ignored.
    """

    mode: str = "bipartite"
    n_src: int = 400
    n_dst: int = 200
    k_src: int = 4
    k_dst: int = 4
    block_matrix: dict = field(
        default_factory=lambda: {"kind": "assortative", "p_in": 1.0, "p_out": 0.15}
    )
    base_rate: float = 100.0  # expected events per unit time
    t_end: float = 100.0  # horizon
    d_node: int = 8
    node_sigma: float = 0.5
    beta_h: float = 0.5
    d_edge: int = 4
    edge_sigma: float = 1.0
    pop_src: dict = field(default_factory=lambda: {"kind": "none"})
    pop_dst: dict = field(default_factory=lambda: {"kind": "none"})
    window_events: int = 2000  # default analysis window w (events)
    seed: int = 0
    # Id layout (default "reserved"; "legacy" is kept for backward compatibility). "legacy": fresh ids from node churn are appended after all initial ids (ranges mix,
    # table size reveals eps). "reserved": each side owns a fixed id range of size
    # ceil((1 + id_headroom) * n) chosen from the config only (never from the schedule); fresh ids
    # are allocated inside the owner's range, and the node table has a fixed size.
    id_layout: str = "reserved"
    id_headroom: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in ("bipartite", "unipartite"):
            raise ValueError("mode must be 'bipartite' or 'unipartite'")
        if self.id_layout not in ("legacy", "reserved"):
            raise ValueError("id_layout must be 'legacy' or 'reserved'")
        if self.mode == "unipartite":
            self.n_dst, self.k_dst = self.n_src, self.k_src

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> DSBMConfig:
        from omegaconf import OmegaConf

        d = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        d.update(overrides)
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    def capacity(self, n: int) -> int:
        """Size of one side's id range under the reserved layout."""
        return int(np.ceil((1.0 + self.id_headroom) * n))

    @property
    def unipartite(self) -> bool:
        return self.mode == "unipartite"


@dataclass
class SynthResult:
    """Generator output. ``stream`` is learner-visible; everything else is ground truth."""

    stream: EventStream
    schedule: DriftSchedule
    resolved: list[dict]
    node_block: np.ndarray  # block of every id (incl. fresh ids); -1 for padding
    cfg: DSBMConfig

    def save(self, directory: str | Path) -> None:
        """Write ``stream.pt``, ``drift_truth.csv`` and ``config.yaml`` into ``directory``."""
        from omegaconf import OmegaConf

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.stream.save(d / "stream.pt")
        self.schedule.to_csv(d / "drift_truth.csv", self.resolved)
        OmegaConf.save(OmegaConf.create(self.cfg.to_dict()), d / "config.yaml")
        np.save(d / "node_block.npy", self.node_block)


# ----------------------------------------------------------------------------------------------
# Construction helpers
# ----------------------------------------------------------------------------------------------


def _block_labels(n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Near-equal blocks: labels (length n, block-contiguous) and boundaries ``starts`` (k+1)."""
    sizes = np.full(k, n // k, dtype=np.int64)
    sizes[: n % k] += 1
    starts = np.concatenate([[0], np.cumsum(sizes)])
    return np.repeat(np.arange(k), sizes), starts


def make_block_matrix(spec: dict, k_src: int, k_dst: int) -> np.ndarray:
    """Relative block intensities. ``assortative``: ``p_in`` on matching blocks, ``p_out`` elsewhere."""
    kind = spec.get("kind", "assortative")
    if kind == "assortative":
        b = np.full((k_src, k_dst), float(spec["p_out"]))
        for i in range(min(k_src, k_dst)):
            b[i, i] = float(spec["p_in"])
        return b
    if kind == "explicit":
        b = np.asarray(spec["matrix"], dtype=float)
        if b.shape != (k_src, k_dst):
            raise ValueError("explicit block matrix has wrong shape")
        return b
    raise ValueError(f"unknown block matrix kind {kind!r}")


def _activity(
    spec: dict, labels: np.ndarray, starts: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    r"""Activity levels $\theta_u$ (degree correction), rescaled so $\Theta_a = n_a$ in every block."""
    n = len(labels)
    kind = spec.get("kind", "none")
    if kind == "none":
        return np.ones(n)
    if kind == "pareto":
        theta = rng.pareto(float(spec["alpha"]), n) + 1.0
    elif kind == "lognormal":
        theta = rng.lognormal(0.0, float(spec["sigma"]), n)
    else:
        raise ValueError(f"unknown popularity kind {kind!r}")
    for a in range(len(starts) - 1):
        s, e = starts[a], starts[a + 1]
        theta[s:e] *= (e - s) / theta[s:e].sum()
    return theta


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


@dataclass
class _Side:
    """Static data for one node set (slots are block-contiguous)."""

    n: int
    k: int
    labels: np.ndarray
    starts: np.ndarray
    theta: np.ndarray
    base_ids: np.ndarray
    base_unit: np.ndarray
    role: int


@dataclass
class _Tables:
    """Per-regime view of a side: current ids / unit features after node churn."""

    ids: np.ndarray
    unit: np.ndarray


def _mean_acceptance(
    src: _Side, dst: _Side, ts: _Tables, td: _Tables, beta_h: float, unipartite: bool
) -> np.ndarray:
    r"""Exact $c_{ab} = \sum_{u\ne v}\theta_u\theta_v e^{\beta_h(\cos-|\beta_h|)}/(\Theta_a\Theta_b)$."""
    out = np.zeros((src.k, dst.k))
    for a in range(src.k):
        sa, ea = src.starts[a], src.starts[a + 1]
        th_a = src.theta[sa:ea]
        for b in range(dst.k):
            sb, eb = dst.starts[b], dst.starts[b + 1]
            th_b = dst.theta[sb:eb]
            if beta_h == 0.0:
                num = th_a.sum() * th_b.sum()
                diag = float(th_a @ th_a) if (unipartite and a == b) else 0.0
            else:
                num = 0.0
                for i in range(sa, ea, 2048):
                    j = min(i + 2048, ea)
                    acc = np.exp(beta_h * (ts.unit[i:j] @ td.unit[sb:eb].T) - abs(beta_h))
                    num += float(src.theta[i:j] @ (acc @ th_b))
                diag = (
                    float(th_a @ th_a) * np.exp(beta_h - abs(beta_h))
                    if (unipartite and a == b)
                    else 0.0
                )
            out[a, b] = (num - diag) / (th_a.sum() * th_b.sum())
    return out


# ----------------------------------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------------------------------


def generate_stream(cfg: DSBMConfig, schedule: DriftSchedule | None = None) -> SynthResult:
    """Simulate the stream and record exact ground truth.

    The result is a deterministic function of ``(cfg, schedule)`` (seed in ``cfg.seed``).
    """
    schedule = schedule or DriftSchedule([])
    events = schedule.events
    ss = np.random.SeedSequence(cfg.seed)
    children = ss.spawn(3 + len(events))
    rng_struct, rng_sim = np.random.default_rng(children[0]), np.random.default_rng(children[1])
    rng_ev = [np.random.default_rng(c) for c in children[3:]]
    uni = cfg.unipartite

    # --- static structure -----------------------------------------------------------------
    n_mu = max(cfg.k_src, cfg.k_dst)
    mu = rng_struct.normal(size=(n_mu, cfg.d_node))
    mu = mu / np.linalg.norm(mu, axis=1, keepdims=True)  # block feature means on the unit sphere

    def build_side(n: int, k: int, id0: int, pop: dict, role: int) -> _Side:
        labels, starts = _block_labels(n, k)
        theta = _activity(pop, labels, starts, rng_struct)
        x = mu[labels] + cfg.node_sigma * rng_struct.normal(size=(n, cfg.d_node))
        return _Side(
            n, k, labels, starts, theta, np.arange(id0, id0 + n, dtype=np.int64), _unit(x), role
        )

    reserved = cfg.id_layout == "reserved"
    cap_s = cfg.capacity(cfg.n_src) if reserved else cfg.n_src
    cap_d = cfg.capacity(cfg.n_dst) if reserved else cfg.n_dst
    src_side = build_side(cfg.n_src, cfg.k_src, 0, cfg.pop_src, ROLE_SRC)
    dst_side = src_side if uni else build_side(cfg.n_dst, cfg.k_dst, cap_s, cfg.pop_dst, ROLE_DST)
    sides = {"src": src_side, "dst": dst_side}
    n_initial = cfg.n_src if uni else cfg.n_src + cfg.n_dst
    B0 = make_block_matrix(cfg.block_matrix, cfg.k_src, cfg.k_dst)
    if uni and not np.allclose(B0, B0.T):
        raise ValueError("unipartite block matrix must be symmetric")
    n_s = np.diff(src_side.starts).astype(float)
    n_d = np.diff(dst_side.starts).astype(float)

    # --- resolve every configured event into concrete seeded quantities ---------------------
    resolved: list[dict] = []  # JSON-able summary written to drift_truth.csv
    res_inj: list[dict] = []  # full objects used by the injectors
    node_feat_extra: list[np.ndarray] = []
    fresh_ids: list[np.ndarray] = []
    node_block_extra: list[np.ndarray] = []
    node_role_extra: list[np.ndarray] = []
    next_id = n_initial  # legacy: one shared counter
    next_by_side = {"src": cfg.n_src, "dst": cap_s + cfg.n_dst}  # reserved: per-side counters
    limit_by_side = {"src": cap_s, "dst": cap_s + cap_d}
    for i, ev in enumerate(events):
        rg = rng_ev[i]
        if ev.type == ALPHA:
            d = rg.normal(size=cfg.d_edge)
            d /= np.linalg.norm(d)
            scale = float(ev.params.get("scale", cfg.edge_sigma * np.sqrt(cfg.d_edge)))
            res_inj.append({"shift": scale * d})
            resolved.append({"shift_norm": scale})
        elif ev.type == BETA:
            kind = ev.params.get("target", "merge")
            opts = {k: v for k, v in ev.params.items() if k not in ("target", "preserve_mass")}
            tgt = beta_target(
                B0, n_s, n_d, kind, rg, bool(ev.params.get("preserve_mass", True)), **opts
            )
            res_inj.append({"B_target": tgt})
            resolved.append({"target": kind})
        elif ev.type == GAMMA:
            side_opt = ev.params.get("side", "src")
            names = (
                ["src"]
                if (uni or side_opt == "src")
                else (["dst"] if side_opt == "dst" else ["src", "dst"])
            )
            per_side, summary = {}, {}
            for name in names:
                sd = sides[name]
                n_max = int(np.floor(ev.eps * sd.n))
                slots = rg.permutation(sd.n)[:n_max]
                if reserved:
                    first = next_by_side[name]
                    if first + n_max > limit_by_side[name]:
                        raise ValueError(
                            f"churn needs more fresh {name} ids than id_headroom={cfg.id_headroom} allows"
                        )
                    next_by_side[name] += n_max
                else:
                    first = next_id
                ids = np.arange(first, first + n_max, dtype=np.int64)
                x = mu[sd.labels[slots]] + cfg.node_sigma * rg.normal(size=(n_max, cfg.d_node))
                per_side[name] = {"slots": slots, "ids": ids, "unit": _unit(x)}
                node_feat_extra.append(x.astype(np.float32))
                node_block_extra.append(sd.labels[slots].astype(np.int64))
                node_role_extra.append(np.full(n_max, sd.role, dtype=np.int8))
                fresh_ids.append(ids)
                summary[name] = {"n_max": n_max, "id_start": int(first)}
                if not reserved:
                    next_id += n_max
            res_inj.append(
                {"sides": names, "n_side": {k: sides[k].n for k in names}, "per_side": per_side}
            )
            resolved.append({"sides": summary})
        else:  # DELTA
            mode = ev.params.get("mode", "permute")
            if mode not in ("permute", "mix"):
                raise ValueError("delta mode must be 'permute' (primary) or 'mix' (sensitivity)")
            if mode == "permute" and (ev.eps != 1.0 or ev.onset != "abrupt"):
                raise ValueError(
                    "delta mode='permute' is a full permutation: use eps=1.0 and an abrupt onset "
                    "(or mode='mix' for a magnitude)"
                )
            # A permutation that leaves B unchanged is a silent no-op (e.g. an assortative
            # unipartite B is invariant under any block relabelling): redraw, else refuse.
            for _attempt in range(50):
                perm = delta_permutation(B0, n_s, n_d, uni, rg, ev.params.get("perm"))
                permuted = B0[perm][:, perm] if uni else B0[:, perm]
                if not np.allclose(permuted, B0) or ev.params.get("perm") is not None:
                    break
            if np.allclose(permuted, B0):
                raise ValueError(
                    "delta would not change B for this block matrix (no-op); use a B with distinct blocks"
                )
            res_inj.append(
                {
                    "perm": perm,
                    "unipartite": uni,
                    "flip_homophily": bool(ev.params.get("flip_homophily", False)),
                    "mode": mode,
                }
            )
            resolved.append(
                {
                    "mode": mode,
                    "perm": perm.tolist(),
                    "flip_homophily": bool(ev.params.get("flip_homophily", False)),
                }
            )

    # --- node tables for every id ever created -------------------------------------------
    # exported node_feat holds the unit-norm features (the ones that enter the homophily term)
    node_feat = np.concatenate(
        [s.base_unit for s in ([src_side] if uni else [src_side, dst_side])]
        + [_unit(f) for f in node_feat_extra],
        axis=0,
    )
    node_block = np.concatenate(
        [s.labels for s in ([src_side] if uni else [src_side, dst_side])] + node_block_extra
    ).astype(np.int64)
    node_role = np.concatenate(
        [np.full(s.n, s.role, dtype=np.int8) for s in ([src_side] if uni else [src_side, dst_side])]
        + node_role_extra
    )
    node_initial = late_ids = late_feat = None
    if reserved:  # fixed-size tables indexed by id; unallocated rows are zero / -1 / not initial
        base_ids = np.concatenate(
            [s.base_ids for s in ([src_side] if uni else [src_side, dst_side])]
        )
        all_ids = np.concatenate([base_ids] + fresh_ids)
        n_tab = cap_s if uni else cap_s + cap_d
        role_tab = np.full(n_tab, ROLE_SRC, dtype=np.int8)
        role_tab[cap_s:] = ROLE_DST  # empty for unipartite
        feat_tab = np.zeros((n_tab, node_feat.shape[1]), dtype=np.float32)
        block_tab = np.full(n_tab, -1, dtype=np.int64)
        feat_tab[all_ids], block_tab[all_ids] = node_feat, node_block
        node_initial = np.zeros(n_tab, dtype=bool)
        node_initial[base_ids] = True
        late_ids = np.flatnonzero(~node_initial & (block_tab >= 0))
        late_feat = feat_tab[late_ids].copy()
        feat_tab[late_ids] = (
            0.0  # fresh entities' features leave the table; StreamReader releases them
        )
        node_feat, node_block, node_role = feat_tab, block_tab, role_tab

    # --- time pieces --------------------------------------------------------------------
    pts = {0.0, float(cfg.t_end)}
    for ev in events:
        pts.update(p for p in ev.breakpoints() if 0.0 < p < cfg.t_end)
    bounds = np.array(sorted(pts))

    base_regime = Regime(B0, cfg.beta_h, np.zeros(cfg.d_edge))
    table_cache: dict[tuple, tuple[_Tables, _Tables]] = {}
    acc_cache: dict[tuple, np.ndarray] = {}

    def regime_at(t: float) -> Regime:
        reg = base_regime
        for i, ev in enumerate(events):
            reg = apply_event(reg, ev, i, ev.ramp(t), res_inj[i])
        return reg

    def tables_for(churn: tuple) -> tuple[_Tables, _Tables]:
        if churn in table_cache:
            return table_cache[churn]
        ts = _Tables(src_side.base_ids.copy(), src_side.base_unit.copy())
        td = ts if uni else _Tables(dst_side.base_ids.copy(), dst_side.base_unit.copy())
        for eid, name, n in churn:
            part = res_inj[eid]["per_side"][name]
            tab = ts if name == "src" else td
            tab.ids[part["slots"][:n]] = part["ids"][:n]
            tab.unit[part["slots"][:n]] = part["unit"][:n]
        table_cache.clear()
        table_cache[churn] = (ts, td)
        return ts, td

    out_src, out_dst, out_t, out_x = [], [], [], []
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        reg = regime_at(0.5 * (t0 + t1))
        ts, td = tables_for(reg.churn)
        key = (reg.churn, reg.beta_h)
        if key not in acc_cache:
            acc_cache.clear()
            acc_cache[key] = _mean_acceptance(src_side, dst_side, ts, td, reg.beta_h, uni)
        c_ab = acc_cache[key]
        theta_blk_s = np.add.reduceat(src_side.theta, src_side.starts[:-1])
        theta_blk_d = np.add.reduceat(dst_side.theta, dst_side.starts[:-1])
        rel = reg.B * np.outer(theta_blk_s, theta_blk_d)
        target = cfg.base_rate * rel / rel.sum()
        counts = rng_sim.poisson(target / c_ab * (t1 - t0))
        seg_src, seg_dst, seg_t = [], [], []
        for a in range(src_side.k):
            sa, ea = src_side.starts[a], src_side.starts[a + 1]
            cum_a = np.cumsum(src_side.theta[sa:ea])
            for b in range(dst_side.k):
                n = int(counts[a, b])
                if n == 0:
                    continue
                sb, eb = dst_side.starts[b], dst_side.starts[b + 1]
                cum_b = np.cumsum(dst_side.theta[sb:eb])
                u = sa + np.minimum(
                    np.searchsorted(cum_a, rng_sim.random(n) * cum_a[-1], side="right"), ea - sa - 1
                )
                v = sb + np.minimum(
                    np.searchsorted(cum_b, rng_sim.random(n) * cum_b[-1], side="right"), eb - sb - 1
                )
                cos = np.einsum("ij,ij->i", ts.unit[u], td.unit[v])
                keep = rng_sim.random(n) < np.exp(reg.beta_h * cos - abs(reg.beta_h))
                if uni:
                    keep &= u != v
                seg_src.append(ts.ids[u[keep]])
                seg_dst.append(td.ids[v[keep]])
                seg_t.append(rng_sim.uniform(t0, t1, size=int(keep.sum())))
        if not seg_t:
            continue
        s_arr, d_arr, t_arr = (
            np.concatenate(seg_src),
            np.concatenate(seg_dst),
            np.concatenate(seg_t),
        )
        order = np.argsort(t_arr, kind="stable")
        marks = reg.mark_mean + cfg.edge_sigma * rng_sim.standard_normal(
            (len(order), cfg.d_edge), dtype=np.float32
        )
        out_src.append(s_arr[order])
        out_dst.append(d_arr[order])
        out_t.append(t_arr[order])
        out_x.append(marks.astype(np.float32))

    stream = EventStream(
        np.concatenate(out_src),
        np.concatenate(out_dst),
        np.concatenate(out_t),
        np.concatenate(out_x),
        node_feat=node_feat,
        node_ids=np.arange(len(node_feat), dtype=np.int64),
        node_role=node_role,
        node_initial=node_initial,
        late_ids=late_ids,
        late_feat=late_feat,
    )
    return SynthResult(stream, schedule, resolved, node_block, cfg)


# ----------------------------------------------------------------------------------------------
# Ground-truth diagnostics (use block labels; for tests and the orthogonality audit only)
# ----------------------------------------------------------------------------------------------


def block_pair_counts(
    src: np.ndarray, dst: np.ndarray, node_block: np.ndarray, k_src: int, k_dst: int
) -> np.ndarray:
    """Matrix of event counts per (source block, destination block)."""
    flat = node_block[src] * k_dst + node_block[dst]
    return np.bincount(flat, minlength=k_src * k_dst).reshape(k_src, k_dst)


def expected_block_counts(cfg: DSBMConfig, n_total: float | None = None) -> np.ndarray:
    r"""Expected event count per block pair in a stationary stream: $\lambda_0 T\,B_{ab}n_an_b/Z$."""
    B = make_block_matrix(cfg.block_matrix, cfg.k_src, cfg.k_dst)
    _, st_s = _block_labels(cfg.n_src, cfg.k_src)
    _, st_d = _block_labels(cfg.n_dst, cfg.k_dst)
    rel = B * np.outer(np.diff(st_s), np.diff(st_d))
    total = cfg.base_rate * cfg.t_end if n_total is None else n_total
    return total * rel / rel.sum()
