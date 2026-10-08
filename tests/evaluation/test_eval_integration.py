"""Integration tests for the standalone evaluation entrypoint.

These cover the two things that are easy to get wrong when decoupling
evaluation from training:

* the checkpoint round-trip -- the backbone and predictive method are excluded
  from ``save_hyperparameters``, so ``load_from_checkpoint`` must be given them
  as kwargs and then restore only the backbone weights;
* the ``run_evaluation`` helper end-to-end against the synthetic depth data.

The Hydra ``eval.yaml`` config is also composed and (partially) instantiated to
guard against config drift.
"""

import math
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluate import run_evaluation
from src.evaluation.metrics import EVAL_METRICS
from src.methods import Diag
from src.models.gaussian_predictive import LightningPredictiveModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _TinyBackbone(torch.nn.Module):
    """A 1x1 conv head so the round-trip carries real (small) weights."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.head = torch.nn.Conv2d(3, out_channels, kernel_size=1)

    def forward(self, x):
        return self.head(x)


def _make_method(out_channels=1):
    return Diag(
        loss="mse",
        eps=1e-9,
        n_train_samples=4,
        n_eval_samples=8,
        out_channels=out_channels,
        gamma=2.0,
    )


def test_checkpoint_round_trip(tmp_path):
    """load_from_checkpoint restores backbone weights given the ignored kwargs."""
    import lightning as L

    method = _make_method()
    backbone = _TinyBackbone(method.out_channels)
    model = LightningPredictiveModel(backbone=backbone, predictive_method=method)

    # Save through Lightning so the checkpoint carries the metadata
    # load_from_checkpoint expects (version, hparams, etc.).
    ckpt = tmp_path / "model.ckpt"
    trainer = L.Trainer(accelerator="cpu", devices=1, logger=False)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(ckpt)

    # Rebuild the (non-checkpointed) backbone + method and reload weights.
    method2 = _make_method()
    backbone2 = _TinyBackbone(method2.out_channels)
    loaded = LightningPredictiveModel.load_from_checkpoint(
        ckpt,
        backbone=backbone2,
        predictive_method=method2,
        map_location="cpu",
    )

    for p, q in zip(model.parameters(), loaded.parameters()):
        assert torch.allclose(p, q)


def test_run_evaluation_end_to_end():
    """run_evaluation returns every metric over a tiny in-memory test set."""
    method = _make_method()
    backbone = _TinyBackbone(method.out_channels)
    model = LightningPredictiveModel(backbone=backbone, predictive_method=method)
    model.eval()

    # A minimal datamodule stub exposing the bits run_evaluation needs.
    class _DM:
        def setup(self, stage=None):
            pass

        def test_dataloader(self, test=False):
            x = torch.randn(5, 3, 8, 10)
            y = torch.randn(5, 1, 8, 10)
            return DataLoader(TensorDataset(x, y), batch_size=2)

    results = run_evaluation(
        forward=model,
        predictive_method=method,
        datamodule=_DM(),
        device="cpu",
    )
    assert set(results) == set(EVAL_METRICS)
    assert all(isinstance(v, float) for v in results.values())


def test_eval_config_composes_and_instantiates(depth_data_dir):
    """eval.yaml composes and its datamodule/method instantiate cleanly."""
    import hydra

    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="eval",
            return_hydra_config=True,
            overrides=["ckpt_path=dummy.ckpt"],
        )
        with open_dict(cfg):
            cfg.paths.root_dir = str(PROJECT_ROOT)
            cfg.datamodule.data_path = depth_data_dir
            cfg.datamodule.num_workers = 0
            cfg.datamodule.eval_batch_size = 4
            cfg.logger = None

        datamodule = hydra.utils.instantiate(cfg.datamodule)
        out_channels = datamodule.get_channels()
        # Bandwidths are resolved per factor as gamma = sqrt(factor * d), mirroring
        # evaluate(); cfg.gamma_eval is a list of factors.
        d = datamodule.get_total_dimension()
        gamma_eval = [math.sqrt(f * d) for f in cfg.gamma_eval]
        method = hydra.utils.instantiate(
            cfg.method, out_channels=out_channels, gamma=gamma_eval, gamma_eval=gamma_eval
        )
        assert method.gamma_list == gamma_eval
        assert method.gamma == gamma_eval[0]
        assert datamodule.eval_batch_size == 4

    GlobalHydra.instance().clear()
