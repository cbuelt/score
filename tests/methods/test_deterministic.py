import math

import torch
import pytest

from src.methods import Deterministic


GAMMA = 0.5
OUT_CHANNELS = 1
BATCH_SIZE = 8
D = 5
EPS = 1e-9
N_TRAIN_SAMPLES = 10
N_EVAL_SAMPLES = 50


def test_out_channels():
    """
    Test correct dimensionality
    """
    approximation = Deterministic("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA)
    out_channels = approximation.out_channels
    assert out_channels == OUT_CHANNELS


@pytest.fixture
def method():
    """A Deterministic instance shared across the metric tests."""
    return Deterministic("mse", EPS, N_TRAIN_SAMPLES, N_EVAL_SAMPLES, OUT_CHANNELS, GAMMA)


def _full(value: float) -> torch.Tensor:
    """A constant tensor with the shape every example uses."""
    return torch.full((BATCH_SIZE, OUT_CHANNELS, D), value)


CASES = {
    "both_zero": dict(
        pred=_full(0.0),
        y=_full(0.0),
        mse=0.0,
        mae=0.0,
        energy_score=0.0,
        kernel_score=0.0,
        crps=0.0,
    ),
    "pred_one": dict(
        pred=_full(1.0),
        y=_full(0.0),
        mse=1.0,
        mae=1.0,
        # ES of a point forecast is the Euclidean residual norm: sqrt(D * 1^2).
        energy_score=math.sqrt(D),
        kernel_score=-torch.exp(-1 / GAMMA**2 * torch.tensor(5.0)).item() + 1.0,
        crps=1.0,
    ),
    "pred_minus_two": dict(
        pred=_full(-2.0),
        y=_full(0.0),
        mse=4.0,
        mae=2.0,
        # ES of a point forecast is the Euclidean residual norm: sqrt(D * 2^2).
        energy_score=math.sqrt(D * 4.0),
        kernel_score=-torch.exp(-1 / GAMMA**2 * torch.tensor(20.0)).item() + 1.0,
        crps=2.0,
    ),
}

CASE_IDS = list(CASES)
CASE_VALUES = list(CASES.values())


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_mse(method, case):
    result = method.mse(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["mse"]))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_mae(method, case):
    result = method.mae(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["mae"]))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_nll(method, case):
    # A deterministic prediction has no density, so the NLL is infinite.
    result = method.nll(case["pred"], case["y"])
    assert result == torch.inf


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_energy_score(method, case):
    result = method.energy_score(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["energy_score"]))


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_kernel_score(method, case):
    result = method.kernel_score(case["pred"], case["y"])
    assert torch.isclose(result, torch.tensor(case["kernel_score"]))
