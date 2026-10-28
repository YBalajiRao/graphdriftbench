"""Drift-schedule builders: abrupt, gradual (ramp), mixed and recurring (A->B->A)."""

from __future__ import annotations

from gdb.synth.injectors import DriftEvent, DriftSchedule


def abrupt(type: str, tau: float, eps: float, **params) -> DriftSchedule:
    """Single instantaneous injection at ``tau``."""
    return DriftSchedule([DriftEvent(tau, type, eps, params)])


def gradual(
    type: str, tau: float, eps: float, ramp_width: float, ramp_steps: int = 10, **params
) -> DriftSchedule:
    """Magnitude ramps linearly from 0 at ``tau`` to ``eps`` at ``tau + ramp_width``.

    The ramp is realised as ``ramp_steps`` piecewise-constant steps of the generator's law.
    """
    return DriftSchedule(
        [
            DriftEvent(
                tau,
                type,
                eps,
                params,
                onset="gradual",
                ramp_width=ramp_width,
                ramp_steps=ramp_steps,
            )
        ]
    )


def mixed(
    types: list[str],
    tau: float,
    eps: float | dict[str, float],
    params: dict[str, dict] | None = None,
) -> DriftSchedule:
    """Simultaneous injections of several types at ``tau`` (``eps`` may be per-type)."""
    params = params or {}
    return DriftSchedule(
        [
            DriftEvent(tau, t, eps[t] if isinstance(eps, dict) else eps, params.get(t, {}))
            for t in types
        ]
    )


def recurring(type: str, tau_on: float, tau_off: float, eps: float, **params) -> DriftSchedule:
    """Regime A -> B -> A: the drift is active on ``[tau_on, tau_off)`` and the original law returns.

    For gamma, "restore" means the departed ids return (their original features and identities).
    """
    return DriftSchedule([DriftEvent(tau_on, type, eps, params, end=tau_off)])
