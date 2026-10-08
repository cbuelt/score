# --- Project root bootstrap (must run before any `src` import) ---
# Walk up to the `.project-root` marker, put the repo root on sys.path so
# `import src...` works when run as `python src/evaluation/evaluate.py`, and
# export PROJECT_ROOT for Hydra's ${oc.env:PROJECT_ROOT} interpolation.
import math
import os
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / ".project-root").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        os.environ.setdefault("PROJECT_ROOT", str(_parent))
        break
# -----------------------------------------------------------------

from typing import Any, Dict, List, Optional

import hydra
import lightning as L
import torch
from lightning import LightningDataModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src.evaluation.evaluator import Evaluator
from src.methods import PredictiveMethod
from src.methods.base import _as_list
from src.models.gaussian_predictive import LightningPredictiveModel
from src.utils import RankedLogger, instantiate_loggers

log = RankedLogger(__name__, rank_zero_only=True)


def metric_kwargs_from_cfg(cfg: DictConfig) -> Dict[str, dict]:
    """Collect per-metric keyword arguments from a config.

    Most metrics take only ``(method, pred, y)``; the ones that are configurable
    read their settings from a config block named after them. Currently just
    ``variogram:`` (see :mod:`src.evaluation.variogram`), whose keys map onto
    ``variogram_score``'s ``weight`` / ``block_size`` parameters. Absent or
    ``null`` entries fall through to the metric's own defaults.
    """
    variogram = cfg.get("variogram")
    if not variogram:
        return {}
    settings = {
        key: value
        for key, value in OmegaConf.to_container(variogram, resolve=True).items()
        if value is not None
    }
    return {"variogram_score": settings} if settings else {}


def run_evaluation(
    forward: Any,
    predictive_method: PredictiveMethod,
    datamodule: LightningDataModule,
    device: str = "cpu",
    metrics: Optional[List[str]] = None,
    loggers: Optional[List[Logger]] = None,
    prefix: str = "test",
    metric_kwargs: Optional[Dict[str, dict]] = None,
) -> Dict[str, float]:
    """Run the decoupled evaluation protocol over the test set.

    Shared by the standalone entrypoint and the in-training eval phase so both
    use a single code path.

    :param forward: callable ``x -> pred`` returning processed predictions
        (e.g. a ``LightningPredictiveModel``).
    :param predictive_method: the method whose metrics / sampler are used.
    :param datamodule: provides ``setup("test")`` and ``test_dataloader``.
    :param device: device to evaluate on.
    :param metrics: metric names to compute; ``None`` -> all registered.
    :param loggers: Lightning loggers to push the results to (prefixed).
    :param prefix: prefix prepended to each metric name when logging.
    :param metric_kwargs: per-metric extra kwargs, see
        :func:`metric_kwargs_from_cfg`.
    :return: ``{metric_name: value}`` (unprefixed).
    """
    datamodule.setup("test")
    evaluator = Evaluator(
        predictive_method, metrics=metrics, device=device, metric_kwargs=metric_kwargs
    )
    results = evaluator.evaluate(forward, datamodule.test_dataloader())

    log.info(f"Evaluation results: {results}")

    if loggers:
        prefixed = {f"evaluation/{prefix}_{name}": value for name, value in results.items()}
        for logger in loggers:
            logger.log_metrics(prefixed)

    return results


def evaluate(cfg: DictConfig) -> Dict[str, float]:
    """Evaluate a trained checkpoint with the decoupled protocol.

    The backbone and predictive method are not stored in the checkpoint (they
    are excluded from ``save_hyperparameters``), so they are rebuilt from config
    and passed to ``load_from_checkpoint``; Lightning then restores only the
    backbone weights from the checkpoint's ``state_dict``.

    :param cfg: a Hydra config (see ``configs/eval.yaml``).
    :return: the metric dictionary.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    data_out_channels: int = datamodule.get_channels()
    data_in_channels: int = datamodule.get_input_channels()

    # Resolve the kernel-score bandwidth(s) as gamma = sqrt(gamma_eval * d),
    # scaling by the target dimension `d` so scores stay roughly
    # dataset-independent under per-coordinate standardization. `gamma_eval` is a list of factors (default
    # [1.0]); multiple factors give a multi-bandwidth score. Standalone eval only
    # ever runs in eval mode, so `gamma_train` is irrelevant here; pass the eval
    # bandwidths for both.
    d = datamodule.get_total_dimension()
    gamma_eval = [math.sqrt(f * d) for f in _as_list(cfg.get("gamma_eval", [1.0]))]
    log.info(f"Resolved gamma_eval from d={d}: {gamma_eval}")

    log.info(f"Instantiating predictive method <{cfg.method._target_}>")
    predictive_method: PredictiveMethod = hydra.utils.instantiate(
        cfg.method,
        out_channels=data_out_channels,
        event_dim=d,
        gamma=gamma_eval,
        gamma_eval=gamma_eval,
    )

    # The backbone must be rebuilt exactly as training built it. `group_denorm` is
    # part of that: PatchTST's RevIN inverse is applied per group, so omitting it
    # silently falls back to the defaults and mis-denormalizes every method whose
    # "other" group is not scale-free (LorD and Chol). `other_zero_init`
    # only affects initialization, but is passed for symmetry with train.py and
    # calibration.py.
    log.info(f"Instantiating backbone <{cfg.backbone._target_}>")
    backbone: torch.nn.Module = hydra.utils.instantiate(
        cfg.backbone,
        in_channels=data_in_channels,
        out_channels=predictive_method.out_channels,
        group_widths=predictive_method.group_widths,
        group_denorm=predictive_method.group_denorm,
        other_zero_init=predictive_method.other_zero_init,
        noise=predictive_method.noise,
    )

    log.info(f"Loading checkpoint <{cfg.ckpt_path}>")
    device = cfg.get("device", "cpu")
    model: LightningPredictiveModel = LightningPredictiveModel.load_from_checkpoint(
        cfg.ckpt_path,
        backbone=backbone,
        predictive_method=predictive_method,
        map_location=device,
    )
    model.eval()
    model.to(device)

    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    results = run_evaluation(
        forward=model,
        predictive_method=predictive_method,
        datamodule=datamodule,
        device=device,
        metrics=cfg.get("metrics"),
        loggers=loggers,
        prefix="test",
        metric_kwargs=metric_kwargs_from_cfg(cfg),
    )
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for standalone evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: None.
    """
    evaluate(cfg)
    return None


if __name__ == "__main__":
    main()
