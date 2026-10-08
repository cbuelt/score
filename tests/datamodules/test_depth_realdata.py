"""Optional smoke test against the *real* depth dataset on disk.

Unlike the other tests (which use a tiny synthetic file), this one reads the
actual ``nyu_train.h5`` from the shared data directory. It is therefore slow
and only works on a machine where that path is mounted -- so it is:

  * marked ``@pytest.mark.realdata`` so it can be excluded with
    ``pytest -m "not realdata"``, and
  * guarded with ``pytest.mark.skipif`` so it is *skipped* (not failed) when the
    data is not present.

A skipped test shows up as ``s`` in the pytest output, not as a failure.
"""

import os

import pytest

from src.datamodules.datasets import DepthDataset
from src.utils.paths import configured_path

REAL_DATA_PATH = os.path.join(configured_path("nyu_dir"), "")
_HAS_REAL_DATA = os.path.exists(os.path.join(REAL_DATA_PATH, "nyu_train.h5"))


@pytest.mark.realdata
@pytest.mark.skipif(not _HAS_REAL_DATA, reason="real depth dataset not available")
def test_real_nyu_train_loads_and_is_normalised():
    """The real file loads, is non-empty, and yields (C,H,W) tensors in [0,1]."""
    # preload=False reads lazily from disk so the test does not pull the whole
    # (large) dataset into RAM.
    ds = DepthDataset(data_path=REAL_DATA_PATH, dataset="nyu", split="train", preload=False)

    assert len(ds) > 0

    x, y = ds[0]
    assert x.shape[0] == 3 and y.shape[0] == 1  # channels first
    assert x.shape[1:] == y.shape[1:]           # image and depth share H, W
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
