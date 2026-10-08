from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    # The model is assembled at runtime from a backbone + predictive method,
    # so log those config pieces rather than a single "model" group.
    hparams["backbone"] = cfg["backbone"]
    hparams["method"] = cfg["method"]
    # Kernel bandwidths. The config holds the per-coordinate scaling factors as
    # lists (gamma_train/gamma_eval, default [1.0]; multiple entries give a
    # multi-bandwidth score); the resolved bandwidths actually used are
    # gamma = sqrt(gamma_* * d) per entry, computed at runtime and stored on the
    # predictive method as _gamma_train/_gamma_eval (both lists). Log both so the
    # run records the knob and the values it produced. There is no top-level
    # `gamma` key.
    hparams["gamma_train"] = cfg.get("gamma_train")
    hparams["gamma_eval"] = cfg.get("gamma_eval")
    predictive_method = getattr(model, "predictive_method", None)
    if predictive_method is not None:
        hparams["gamma_train_resolved"] = getattr(
            predictive_method, "_gamma_train", None
        )
        hparams["gamma_eval_resolved"] = getattr(
            predictive_method, "_gamma_eval", None
        )

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")

    hparams["task_name"] = cfg.get("task_name")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)