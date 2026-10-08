"""Tests for ``src.datamodules.datasets.DepthDataset``.

How pytest works (quick tour):
- Any function named ``test_*`` in a file named ``test_*.py`` is collected and
  run automatically. No class or registration needed.
- A test "passes" if it runs without raising; it "fails" if an ``assert``
  statement is false or any exception escapes. So a test is just: build a
  situation, then ``assert`` what must be true about it.
- Function arguments whose names match a fixture (here ``depth_data_dir`` from
  conftest.py) are filled in by pytest -- this is "dependency injection" and is
  how tests get their setup data.

What we verify for ``DepthDataset``: the parts most likely to silently break a
training run -- the file is found, the data has the shape the model expects,
and the values are normalised into [0, 1].
"""

import torch

from src.datamodules.datasets import DepthDataset

# Geometry must match the synthetic fixture in conftest.py.
N_SAMPLES = 6
HEIGHT = 8
WIDTH = 10


def test_len_matches_number_of_samples(depth_data_dir):
    """``len(dataset)`` should report one entry per stored frame."""
    ds = DepthDataset(data_path=depth_data_dir, dataset="nyu", split="train", preload=False)
    assert len(ds) == N_SAMPLES


def test_getitem_returns_channel_first_tensors(depth_data_dir):
    """A sample is (image, depth) with channels first: (C, H, W).

    This guards the rename+transpose logic in ``DepthDataset`` -- the on-disk
    layout is (H, W, C) but models expect (C, H, W). If that transpose ever
    regresses, this test fails.
    """
    ds = DepthDataset(data_path=depth_data_dir, dataset="nyu", split="train", preload=False)
    x, y = ds[0]

    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape == (3, HEIGHT, WIDTH)  # RGB image
    assert y.shape == (1, HEIGHT, WIDTH)  # single-channel depth


def test_getitem_normalises_to_unit_range(depth_data_dir):
    """Values are uint8 divided by 255, so they must land in [0, 1] floats."""
    ds = DepthDataset(data_path=depth_data_dir, dataset="nyu", split="train", preload=False)
    x, y = ds[0]

    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0


def test_filename_is_built_with_underscore(depth_data_dir):
    """The loader must look for ``nyu_train.h5`` (dataset + '_' + split + .h5).

    Guards against concatenating without the underscore (``nyutrain.h5``),
    which would never find the real files.
    """
    ds = DepthDataset(data_path=depth_data_dir, dataset="nyu", split="train", preload=False)
    assert ds.filename == "nyu_train.h5"


def _write_h5(path, image, depth, **extra):
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset("image", data=image)
        f.create_dataset("depth", data=depth)
        for name, data in extra.items():
            f.create_dataset(name, data=data)


def test_unknown_dtypes_are_rejected(tmp_path):
    """Unsupported encodings must fail loudly at construction.

    Two encodings are supported: uint8 (/255) and uint16 millimeter depth
    (BTS-format files, /10000). A float depth or a non-uint8 image would
    otherwise be silently mis-scaled.
    """
    import numpy as np
    import pytest

    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(2, 8, 10, 3), dtype=np.uint8)

    _write_h5(tmp_path / "nyu_train.h5", image, rng.random((2, 8, 10, 1), dtype=np.float32))
    with pytest.raises(TypeError, match="depth"):
        DepthDataset(
            data_path=str(tmp_path) + "/", dataset="nyu", split="train", preload=False
        )

    _write_h5(
        tmp_path / "nyu_test.h5",
        image.astype(np.float32),
        rng.integers(0, 255, size=(2, 8, 10, 1), dtype=np.uint8),
    )
    with pytest.raises(TypeError, match="image"):
        DepthDataset(
            data_path=str(tmp_path) + "/", dataset="nyu", split="test", preload=False
        )


def test_bts_uint16_depth_is_scaled_to_meters_over_ten(tmp_path):
    """The BTS-format files store depth as uint16 millimeters (cap 10 m): samples
    must come back as float32 = mm/10000 in [0, 1], images as /255, and the
    extra ``valid`` variable must not disturb loading."""
    import numpy as np

    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(3, 8, 10, 3), dtype=np.uint8)
    depth = rng.integers(0, 10001, size=(3, 8, 10, 1), dtype=np.uint16)
    valid = rng.integers(0, 2, size=(3, 8, 10, 1), dtype=np.uint8)
    _write_h5(tmp_path / "nyu_bts_train.h5", image, depth, valid=valid)

    ds = DepthDataset(
        data_path=str(tmp_path) + "/", dataset="nyu_bts", split="train", preload=False
    )
    x, y = ds[1]
    assert x.shape == (3, 8, 10) and y.shape == (1, 8, 10)
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    import numpy as np

    assert np.allclose(y.numpy()[0], depth[1, ..., 0].astype(np.float32) / 10000.0)
    assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0


def test_preload_and_lazy_yield_identical_samples(depth_data_dir):
    """``preload=True`` (all in RAM) and ``preload=False`` (lazy from disk)
    must return exactly the same tensors -- preload is purely a memory/speed
    trade-off, not a behavioural change.
    """
    eager = DepthDataset(data_path=depth_data_dir, dataset="nyu", split="train", preload=True)
    lazy = DepthDataset(data_path=depth_data_dir, dataset="nyu", split="train", preload=False)

    assert len(eager) == len(lazy)
    xe, ye = eager[0]
    xl, yl = lazy[0]
    assert torch.equal(xe, xl)
    assert torch.equal(ye, yl)
