from dataclasses import replace

import numpy as np
import pytest

from gdb.data.stream import EventStream
from gdb.synth import schedules as sch
from gdb.synth.dsbm import generate_stream


@pytest.mark.slow
def test_prequential_loss_is_causal_in_events_and_node_features(tiny_cfg):
    """Altering FUTURE events, future entities' features and the schedule must not change past losses."""
    from gdb.models.tgn_prequential import prequential_event_losses

    cfg = replace(tiny_cfg, t_end=40.0, base_rate=100.0)
    a = generate_stream(cfg, sch.abrupt("gamma", 20.0, 0.3, side="both")).stream
    k = 2000  # batch aligned (batch_size 200)
    rng = np.random.default_rng(0)
    perm = k + rng.permutation(len(a) - k)
    b = EventStream(
        np.r_[a.src[:k], a.src[perm]],
        np.r_[a.dst[:k], a.dst[perm]],
        a.t,
        np.r_[a.edge_feat[:k], a.edge_feat[perm]],
        a.node_feat,
        a.node_ids,
        a.node_role,
        a.node_initial,
        a.late_ids,
        a.late_feat + 10.0,  # different features for entities that appear later
    )
    la = prequential_event_losses(a, seed=3)["loss"]
    lb = prequential_event_losses(b, seed=3)["loss"]
    assert np.allclose(la[:k], lb[:k], atol=1e-4)
    assert not np.allclose(la[k:], lb[k:], atol=1e-3)  # the test can fail: the future does differ
