r"""Type-pure drift injectors for the dynamic SBM (Blueprint §4, §6.1).

A *regime* is the generator's law at one instant: block-intensity matrix ``B``, homophily
``beta_h``, edge-mark mean, and node churn (which slots were re-labelled with fresh ids). Each
injector maps a regime to a new regime and changes exactly one factor of
$P(V)P(A\mid V)P(X\mid A,V)P(Y\mid A,X,V)$:

* ``alpha``: edge-mark mean $\mu_e \to \mu_e + \varepsilon\,s\,\mathbf d$ (structure untouched);
* ``beta``: $B \to (1-\varepsilon)B + \varepsilon B'$ (marks and $\beta_h$ untouched);
* ``gamma``: a fraction $\varepsilon$ of nodes get fresh ids (same block, same activity, fresh
  features from the block law); old ids stop emitting;
* ``delta`` (primary, ``mode="permute"``, Blueprint §6.1): $B \to BP^\top$ (bipartite; $PBP^\top$
  unipartite) for a permutation $P$ of blocks with equal size and equal margin, and optionally
  $\beta_h \to -\beta_h$. It has no magnitude: ``eps`` must be 1 and the onset abrupt.
  Secondary sensitivity perturbation (``mode="mix"``): $B \to (1-\varepsilon)B + \varepsilon BP^\top$,
  $\beta_h \to (1-2\varepsilon)\beta_h$. Only the full permutation preserves the spectrum of $B$.

``eps`` is the injection magnitude; ``r(t) \in [0,1]`` is the onset ramp, so the effective
magnitude is ``r(t) * eps``. Nothing here is visible to a learner.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

ALPHA, BETA, GAMMA, DELTA = "alpha", "beta", "gamma", "delta"
TYPES = (ALPHA, BETA, GAMMA, DELTA)


@dataclass(frozen=True)
class DriftEvent:
    """One configured injection, written verbatim to ``drift_truth.csv``.

    Attributes:
        tau: onset time. ``eps``: magnitude (meaning per type, see module docstring).
        params: type-specific options (``side`` for gamma, ``target`` for beta, ...).
        onset: ``"abrupt"`` or ``"gradual"``; gradual ramps r(t) from 0 to 1 over ``ramp_width``
            in ``ramp_steps`` piecewise-constant steps.
        end: if set, the original law is restored at this time (recurring regimes A->B->A).
    """

    tau: float
    type: str
    eps: float
    params: dict = field(default_factory=dict)
    onset: str = "abrupt"
    ramp_width: float = 0.0
    ramp_steps: int = 10
    end: float | None = None

    def __post_init__(self) -> None:
        if self.type not in TYPES:
            raise ValueError(f"unknown drift type {self.type!r}")
        if self.onset not in ("abrupt", "gradual"):
            raise ValueError("onset must be 'abrupt' or 'gradual'")
        if self.onset == "gradual" and self.ramp_width <= 0:
            raise ValueError("gradual onset needs ramp_width > 0")
        if self.end is not None and self.end <= self.tau:
            raise ValueError("end must be after tau")

    def ramp(self, t: float) -> float:
        """Onset ramp $r(t)\\in[0,1]$ (0 before ``tau`` and from ``end`` on)."""
        if t < self.tau or (self.end is not None and t >= self.end):
            return 0.0
        if self.onset == "abrupt":
            return 1.0
        return float(min(1.0, (t - self.tau) / self.ramp_width))

    def breakpoints(self) -> list[float]:
        """Times where the piecewise-constant regime may change."""
        pts = [self.tau]
        if self.onset == "gradual":
            pts += [
                self.tau + self.ramp_width * j / self.ramp_steps
                for j in range(1, self.ramp_steps + 1)
            ]
        if self.end is not None:
            pts.append(self.end)
        return pts


@dataclass
class DriftSchedule:
    """An ordered list of :class:`DriftEvent` (mixed drift = several events with the same tau)."""

    events: list[DriftEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.events = sorted(self.events, key=lambda e: e.tau)

    def __add__(self, other: DriftSchedule) -> DriftSchedule:
        return DriftSchedule(self.events + other.events)

    def __len__(self) -> int:
        return len(self.events)

    def to_csv(self, path: str | Path, resolved: list[dict] | None = None) -> None:
        """Write the ground-truth table ``drift_truth.csv`` (one row per event)."""
        cols = [
            "event_id",
            "tau",
            "type",
            "eps",
            "onset",
            "ramp_width",
            "ramp_steps",
            "end",
            "params",
            "resolved",
        ]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for i, e in enumerate(self.events):
                w.writerow(
                    {
                        "event_id": i,
                        "tau": repr(e.tau),
                        "type": e.type,
                        "eps": repr(e.eps),
                        "onset": e.onset,
                        "ramp_width": repr(e.ramp_width),
                        "ramp_steps": e.ramp_steps,
                        "end": "" if e.end is None else repr(e.end),
                        "params": json.dumps(e.params, sort_keys=True),
                        "resolved": json.dumps(resolved[i] if resolved else {}, sort_keys=True),
                    }
                )

    @classmethod
    def from_csv(cls, path: str | Path) -> DriftSchedule:
        """Read a ``drift_truth.csv`` back into a schedule (the ``resolved`` column is ignored)."""
        events = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                events.append(
                    DriftEvent(
                        tau=float(row["tau"]),
                        type=row["type"],
                        eps=float(row["eps"]),
                        params=json.loads(row["params"]),
                        onset=row["onset"],
                        ramp_width=float(row["ramp_width"]),
                        ramp_steps=int(row["ramp_steps"]),
                        end=float(row["end"]) if row["end"] else None,
                    )
                )
        return cls(events)


@dataclass(frozen=True)
class Regime:
    """Generator law at one instant. ``churn`` lists ``(event_idx, side, n_replaced)``."""

    B: np.ndarray
    beta_h: float
    mark_mean: np.ndarray
    churn: tuple[tuple[int, str, int], ...] = ()


# ----------------------------------------------------------------------------------------------
# Resolution: turn a configured event into concrete (seeded) quantities
# ----------------------------------------------------------------------------------------------


def _weighted_margins(
    B: np.ndarray, n_src: np.ndarray, n_dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Expected-rate margins: row sums $\\sum_b B_{ab} n_b$ and column sums $\\sum_a B_{ab} n_a$."""
    return B @ n_dst, B.T @ n_src


def mass(B: np.ndarray, n_src: np.ndarray, n_dst: np.ndarray) -> float:
    """Total relative event mass $\\sum_{ab} B_{ab} n_a n_b$."""
    return float(n_src @ B @ n_dst)


def beta_target(
    B: np.ndarray,
    n_src: np.ndarray,
    n_dst: np.ndarray,
    kind: str,
    rng: np.random.Generator,
    preserve_mass: bool,
    **opts,
) -> np.ndarray:
    """Target matrix $B'$ for beta drift.

    ``hub``: block 0 on each side becomes a hub (degree concentration: row 0 and column 0 are scaled
    by ``boost``, default 4, so entry [0, 0] by ``boost**2``);
    ``merge``: the first two blocks on each side lose their distinction (average of the 2x2 corner);
    ``densify``: elementwise square (sharper communities); ``random``: i.i.d. Gamma(1) entries.
    With ``preserve_mass`` the target is rescaled to the same total event mass as ``B``, so beta
    changes where events go, not how many.
    """
    if kind == "hub":
        # degree concentration: block 0 on each side becomes a hub (rows/cols scaled by `boost`)
        t = B.copy()
        t[0, :] *= float(opts.get("boost", 4.0))
        t[:, 0] *= float(opts.get("boost", 4.0))
    elif kind == "merge":
        t = B.copy()
        if B.shape[0] < 2 or B.shape[1] < 2:
            raise ValueError("merge needs at least 2 blocks on each side")
        t[0:2, 0:2] = B[0:2, 0:2].mean()
    elif kind == "densify":
        t = B ** float(opts.get("power", 2.0))
    elif kind == "random":
        t = rng.gamma(1.0, size=B.shape)
    else:
        raise ValueError(f"unknown beta target {kind!r}")
    if preserve_mass:
        t = t * (mass(B, n_src, n_dst) / mass(t, n_src, n_dst))
    return t


def delta_permutation(
    B: np.ndarray,
    n_src: np.ndarray,
    n_dst: np.ndarray,
    unipartite: bool,
    rng: np.random.Generator,
    perm: list[int] | None,
) -> np.ndarray:
    """Block permutation for delta: only blocks with equal size and equal margin may be exchanged.

    Bipartite: destination blocks with equal size and equal weighted column sum (so source degrees
    and destination degrees are both preserved). Unipartite: blocks with equal size and equal
    weighted row sum. Returns ``perm`` with ``B' = B[:, perm]`` (bipartite) or ``B[perm][:, perm]``.
    """
    rows, cols = _weighted_margins(B, n_src, n_dst)
    key_margin = rows if unipartite else cols
    sizes = n_src if unipartite else n_dst
    k = len(sizes)
    groups: dict[tuple, list[int]] = {}
    for b in range(k):
        groups.setdefault(
            (int(sizes[b]), round(float(key_margin[b]) / max(key_margin.max(), 1e-12), 6)), []
        ).append(b)
    if perm is not None:
        perm_arr = np.asarray(perm, dtype=np.int64)
        for b in range(k):
            if (int(sizes[b]), round(float(key_margin[b]) / max(key_margin.max(), 1e-12), 6)) != (
                int(sizes[perm_arr[b]]),
                round(float(key_margin[perm_arr[b]]) / max(key_margin.max(), 1e-12), 6),
            ):
                raise ValueError("delta perm exchanges blocks with unequal size or margin")
        return perm_arr
    perm_arr = np.arange(k)
    movable = [g for g in groups.values() if len(g) >= 2]
    if not movable:
        raise ValueError("no pair of blocks with equal size and equal margin: delta cannot be pure")
    for g in movable:
        shift = int(rng.integers(1, len(g)))
        for i, b in enumerate(g):
            perm_arr[b] = g[(i + shift) % len(g)]
    return perm_arr


# ----------------------------------------------------------------------------------------------
# Injectors: regime -> regime
# ----------------------------------------------------------------------------------------------


def inject_alpha(reg: Regime, ev: DriftEvent, r: float, res: dict) -> Regime:
    r"""Feature drift: $\mu_e \to \mu_e + r\varepsilon\,\mathbf s$ with $\mathbf s$ = ``res['shift']``.

    $B$, $\beta_h$ and the node sets are untouched, so only $P(X\mid A,V)$ changes.
    """
    return replace(reg, mark_mean=reg.mark_mean + r * ev.eps * res["shift"])


def inject_beta(reg: Regime, ev: DriftEvent, r: float, res: dict) -> Regime:
    r"""Topological drift: $B \to (1-m)B + mB'$, $m = r\varepsilon$. Marks and $\beta_h$ untouched."""
    m = r * ev.eps
    return replace(reg, B=(1.0 - m) * reg.B + m * res["B_target"])


def inject_gamma(reg: Regime, ev: DriftEvent, r: float, res: dict, event_idx: int) -> Regime:
    r"""Entity turnover: re-label $\lfloor r\varepsilon N\rfloor$ nodes per affected side with fresh ids.

    The replaced slots keep block and activity level; only identity and features change. The
    realised churn level is stored in the regime (``churn``) and applied by the generator.
    """
    added = tuple(
        (event_idx, side, int(np.floor(r * ev.eps * res["n_side"][side]))) for side in res["sides"]
    )
    return replace(reg, churn=reg.churn + added)


def inject_delta(reg: Regime, ev: DriftEvent, r: float, res: dict) -> Regime:
    r"""Relational-semantic drift: $B \to (1-m)B + mBP^\top$ (bipartite) / $PBP^\top$ (unipartite).

    ``permute`` mode: $m=1$ whenever the injection is active (full permutation). ``mix`` mode:
    $m=r\varepsilon$. Block sizes and margins are preserved by $P$, so degree laws are unchanged
    while *which* block pairs interact changes; the spectrum of $B$ is unchanged only for $m=1$.
    With ``flip_homophily``: $\beta_h \to (1-2m)\beta_h$.
    """
    m = 1.0 if res.get("mode", "permute") == "permute" else r * ev.eps
    perm = res["perm"]
    permuted = reg.B[perm][:, perm] if res["unipartite"] else reg.B[:, perm]
    beta_h = (1.0 - 2.0 * m) * reg.beta_h if res["flip_homophily"] else reg.beta_h
    return replace(reg, B=(1.0 - m) * reg.B + m * permuted, beta_h=beta_h)


def apply_event(reg: Regime, ev: DriftEvent, event_idx: int, r: float, res: dict) -> Regime:
    """Dispatch to the injector for ``ev.type`` (``r = 0`` leaves the regime unchanged)."""
    if r <= 0.0:
        return reg
    if ev.type == ALPHA:
        return inject_alpha(reg, ev, r, res)
    if ev.type == BETA:
        return inject_beta(reg, ev, r, res)
    if ev.type == GAMMA:
        return inject_gamma(reg, ev, r, res, event_idx)
    return inject_delta(reg, ev, r, res)
