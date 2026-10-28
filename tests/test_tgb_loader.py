import os

import pytest

from gdb.data import tgb_loader


def test_context_manager_restores_proj_dir_even_on_error(tmp_path):
    import tgb.linkproppred.dataset as m

    original = m.PROJ_DIR
    with tgb_loader.tgb_root(tmp_path) as root:
        assert m.PROJ_DIR == str(root) + os.sep and m.PROJ_DIR != original
    assert m.PROJ_DIR == original
    with pytest.raises(RuntimeError), tgb_loader.tgb_root(tmp_path):
        raise RuntimeError("boom")
    assert m.PROJ_DIR == original


def test_default_root_comes_from_config():
    assert str(tgb_loader.default_data_root()).lower().endswith("tgb_data")


@pytest.mark.slow
def test_load_tgbl_wiki_counts_and_masks():
    import numpy as np

    d = tgb_loader.load_tgb("tgbl-wiki")
    s = d.stream
    assert len(s) == 157_474
    assert len(np.unique(s.src)) == 8_227 and len(np.unique(s.dst)) == 1_000
    assert s.edge_feat.shape == (157_474, 172) and s.edge_feat.dtype == np.float32
    assert s.src.max() < s.dst.min()  # TGB convention: destinations numbered after sources
    assert s.node_initial is None and s.late_ids is None  # loader output unchanged by id_layout
    assert s.src.max() < 8_227 <= s.dst.min() and s.dst.max() < 9_227  # dense TGB ids
    assert np.all(np.diff(s.t) >= 0)
    n = len(s)
    assert (d.train_mask.sum(), d.val_mask.sum(), d.test_mask.sum()) == (110_232, 23_621, 23_621)
    assert (d.train_mask.astype(int) + d.val_mask + d.test_mask == 1).all() and len(
        d.train_mask
    ) == n
    import tgb.linkproppred.dataset as m

    assert "tgb_data" not in m.PROJ_DIR  # patch was undone after loading
