"""The learner side (src/gdb/cl, src/gdb/detect) must never see ground truth."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "gdb"
LEARNER_DIRS = ["cl", "detect"]
FORBIDDEN = [
    r"drift_truth",  # the ground-truth file
    r"\bDriftSchedule\b|\bDriftEvent\b",  # configured schedule objects
    r"\bnode_block\b",  # generator block labels
    r"gdb\.synth",  # nothing from the generator package
    r"\bSynthResult\b",
    r"\bsynth\b",  # also catches `from gdb import synth` and relative imports
    r"\bresolved\b|config\.yaml|data/synth",
]


def _py_files():
    for d in LEARNER_DIRS:
        yield from (SRC / d).rglob("*.py")


def test_learner_packages_exist():
    for d in LEARNER_DIRS:
        assert (SRC / d).is_dir()


def test_learner_code_never_reads_ground_truth():
    hits = []
    for f in _py_files():
        text = f.read_text(encoding="utf-8")
        for pat in FORBIDDEN:
            for m in re.finditer(pat, text):
                hits.append(f"{f.relative_to(SRC)}: {m.group(0)}")
    assert not hits, hits


def test_the_check_itself_can_fail(tmp_path):
    """Guard against a vacuous test: the patterns must match what they are meant to catch."""
    bad = 'df = pd.read_csv("data/x/drift_truth.csv")\nfrom gdb.synth.injectors import DriftEvent\n'
    assert all(re.search(p, bad) for p in (FORBIDDEN[0], FORBIDDEN[1], FORBIDDEN[3]))
