# --- Project root bootstrap (must run before any `src` import) ---
# Walk up to the `.project-root` marker, put the repo root on sys.path so
# `import src...` works when run as `python src/train/train.py`, and export
# PROJECT_ROOT for Hydra's ${oc.env:PROJECT_ROOT} interpolation.
import math
import os
import sys
from pathlib import Path

import wandb

for _parent in Path(__file__).resolve().parents:
    if (_parent / ".project-root").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        os.environ.setdefault("PROJECT_ROOT", str(_parent))
        break
# -----------------------------------------------------------------

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning import Callback, LightningDataModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.evaluation.evaluate import metric_kwargs_from_cfg, run_evaluation
from src.methods import PredictiveMethod
from src.methods.base import _as_list
from src.models.gaussian_predictive import (
    LightningPredictiveModel,
    StagedLightningModel,
)
from src.utils.legacy import METHOD_SLUGS
from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _method_facets(cfg: DictConfig) -> dict:
    """Facets of the method config that distinguish otherwise identical runs.

    Returns a dict with entries present only when the method defines the
    corresponding option: ``transform`` (the basis kind of the correlation
    core, ``"fourier"`` for the native FFT path), ``rank`` (number of low-rank
    core factors) and ``activation`` (the normalized-core activation). Used to
    build the wandb run name and tags, so that ablations differing only in
    these options are distinguishable in the UI.
    """
    method = cfg.get("method")
    facets: dict = {}
    if method is None:
        return facets
    if "transform" in method:
        transform = method.get("transform")
        if transform is None:
            facets["transform"] = "fourier"
        else:
            kind = transform.get("kind", None)
            if kind is None:
                facets["transform"] = "custom"
            elif isinstance(kind, str):
                facets["transform"] = kind
            else:
                facets["transform"] = "-".join(str(k) for k in kind)
    if "rank" in method:
        facets["rank"] = int(method.get("rank") or 0)
    if "core_activation" in method:
        facets["activation"] = str(method.get("core_activation"))
    return facets


# Values that are the config default for their facet and therefore carry no
# information in a run name. Both are still tagged, so either side of an
# ablation stays selectable in the wandb UI.
#
# Must match the stage-1 default in configs/method/score.yaml; otherwise every
# staged run picks up a spurious `s1<loss>` name part (the naming tests check
# this).
_DEFAULT_STAGE1_LOSS = "crps"
_DEFAULT_FEATURES = "M"


def _run_loss_name(cfg: DictConfig) -> str:
    """The loss identifying the run: stage-2 loss for staged runs, else loss.

    In the two-stage fit the stage-2 (core) loss is the informative name
    component. A stage-1 loss deviating from the default is carried
    separately by :func:`_stage1_name_part`, so the two are distinguishable.
    """
    if cfg.get("staged") and cfg.method.get("stage2_loss"):
        return str(cfg.method.stage2_loss)
    return str(cfg.method.get("loss", "loss"))


def _stage1_name_part(cfg: DictConfig) -> Optional[str]:
    """``s1<loss>`` when a staged run deviates from the default stage-1 loss.

    Both stage-1 losses fit the same marginals, so without this the two
    variants of one configuration would receive identical run names.
    """
    if not cfg.get("staged"):
        return None
    stage1 = cfg.method.get("stage1_loss")
    if stage1 is None or str(stage1) == _DEFAULT_STAGE1_LOSS:
        return None
    return f"s1{stage1}"


def _features_name_part(cfg: DictConfig) -> Optional[str]:
    """``uni`` for a univariate time-series run (``features: S``).

    The datamodule choice is the same for both, so univariate and multivariate
    runs of one dataset would otherwise be indistinguishable by name.
    """
    features = cfg.get("datamodule", {}).get("features")
    if features is None or str(features) == _DEFAULT_FEATURES:
        return None
    return "uni" if str(features) == "S" else str(features).lower()


def _method_slug(choices) -> str:
    """The method's slug in run names and tags.

    Deliberately the *pre-release* slug (e.g. ``whitened_spectral_diag`` for
    SCORE, ``phys_lora`` for LorD; see :data:`src.utils.legacy.METHOD_SLUGS`),
    so that new runs keep matching the published results, tables and notebooks.
    Method config names without a mapping are used as-is.
    """
    method = choices.get("method", "method")
    return METHOD_SLUGS.get(method, method)


def _build_run_name(cfg: DictConfig) -> str:
    """Builds a descriptive run name from the dataset, method and loss.

    Format:
    ``<dataset>[_uni]_<method>[_<transform>][_lora<r>][_<activation>][_s1<loss>]_<loss>``,
    where dataset is the Hydra datamodule choice and method is the
    pre-release slug of the method choice (:func:`_method_slug`). The facet
    parts appear only when they carry information: ``uni`` for a univariate
    time-series split, the transform kind when it is not already part of the
    method choice name, ``lora<r>`` for rank > 0, the core activation when it
    deviates from the softmax default, and ``s1<loss>`` when the stage-1 loss
    deviates from its default.
    """
    choices = HydraConfig.get().runtime.choices
    dataset = choices.get("datamodule", "dataset")
    method = _method_slug(choices)
    facets = _method_facets(cfg)
    parts = [dataset]
    features_part = _features_name_part(cfg)
    if features_part:
        parts.append(features_part)
    parts.append(method)
    transform = facets.get("transform")
    if transform and transform != "fourier" and transform not in method:
        parts.append(transform)
    if facets.get("rank"):
        # "lora<r>" as a facet (low-rank correction of the core); a method whose
        # slug already says lora (LorD, published as `phys_lora`) gets the bare
        # rank "r<r>" instead, so sweeps stay distinguishable without stuttering.
        parts.append(f"r{facets['rank']}" if "lora" in method else f"lora{facets['rank']}")
    if facets.get("activation") and facets["activation"] != "softmax":
        parts.append(facets["activation"])
    stage1_part = _stage1_name_part(cfg)
    if stage1_part:
        parts.append(stage1_part)
    parts.append(_run_loss_name(cfg))
    return "_".join(parts)


def _build_wandb_tags(cfg: DictConfig) -> List[str]:
    """One filterable wandb tag per run facet.

    Always tagged: dataset, method (its pre-release slug, see
    :func:`_method_slug`) and the loss(es); staged runs get separate
    ``stage1:``/``stage2:`` tags. Method facets follow the same policy as the
    run name, except that ``transform:`` and ``act:`` are tagged whenever the
    method defines them (so both sides of an ablation are selectable), while
    ``rank:`` is tagged only for rank > 0.
    """
    choices = HydraConfig.get().runtime.choices
    tags = [
        f"dataset:{choices.get('datamodule', 'dataset')}",
        f"method:{_method_slug(choices)}",
    ]
    if cfg.get("staged"):
        if cfg.method.get("stage1_loss"):
            tags.append(f"stage1:{cfg.method.stage1_loss}")
        if cfg.method.get("stage2_loss"):
            tags.append(f"stage2:{cfg.method.stage2_loss}")
    else:
        tags.append(f"loss:{cfg.method.get('loss', 'loss')}")
    facets = _method_facets(cfg)
    if "transform" in facets:
        tags.append(f"transform:{facets['transform']}")
    if facets.get("rank"):
        tags.append(f"rank:{facets['rank']}")
    if "activation" in facets:
        tags.append(f"act:{facets['activation']}")
    features = cfg.get("datamodule", {}).get("features")
    if features is not None:
        tags.append(f"features:{features}")
    if cfg.get("seed") is not None:
        tags.append(f"seed:{cfg.seed}")
    return tags


def _apply_wandb_naming(cfg: DictConfig) -> None:
    """Give the wandb run a descriptive name and filterable facet tags.

    An explicitly configured ``logger.wandb.name`` is left untouched;
    user-supplied ``logger.wandb.tags=[...]`` are kept and extended with the
    generated facet tags.
    """
    if not (cfg.get("logger") and cfg.logger.get("wandb")):
        return
    if not cfg.logger.wandb.get("name"):
        cfg.logger.wandb.name = _build_run_name(cfg)
    existing = [str(t) for t in (cfg.logger.wandb.get("tags") or [])]
    cfg.logger.wandb.tags = existing + [
        t for t in _build_wandb_tags(cfg) if t not in existing
    ]


def _warmup_epochs(scheduler_cfg: Any) -> int:
    """Number of warmup epochs configured on a scheduler partial.

    Warmup is *subtractive*: these epochs are spent inside the configured
    max_epochs budget, so a fit with ``max_epochs=100`` and ``warmup_epochs=3``
    runs 3 warmup epochs followed by 97 cosine epochs -- the configured number is
    the total number of epochs actually run. The trainer's max_epochs is
    therefore used as-is; only the scheduler splits it (see
    _build_optimizer_and_scheduler). Returns 0 when the partial has no
    warmup_epochs (e.g. a scheduler that does not warm up).

    :param scheduler_cfg: an instantiated scheduler partial (functools.partial),
        or None.
    """
    if scheduler_cfg is None:
        return 0
    keywords = getattr(scheduler_cfg, "keywords", {})
    return int(keywords.get("warmup_epochs", 0))


def _log_warmup_budget(
    warmup_epochs: int, max_epochs: Optional[int], label: str
) -> None:
    """Log how a fit's epoch budget splits into warmup and decay epochs.

    Warmup is subtractive (see :func:`_warmup_epochs`), so this only reports the
    split -- it never changes the budget. Warns when the warmup would swallow the
    whole budget, which leaves no epochs for the decay. The decay phase is
    whatever the configured scheduler does after its ramp-up (cosine annealing,
    a one-cycle anneal, a staircase decay), so it is named generically here.

    :param warmup_epochs: configured linear-warmup epochs for this fit.
    :param max_epochs: the fit's total epoch budget, or None when uncapped.
    :param label: what the budget belongs to, for the log message.
    """
    if not warmup_epochs or max_epochs is None:
        return
    if warmup_epochs >= max_epochs:
        log.warning(
            f"{label}: warmup_epochs={warmup_epochs} covers the entire "
            f"max_epochs={max_epochs} budget -- the LR never anneals."
        )
        return
    log.info(
        f"{label}: {max_epochs} epochs = {warmup_epochs} warmup + "
        f"{max_epochs - warmup_epochs} decay."
    )


def _resolve_gammas(
    cfg: DictConfig, datamodule: LightningDataModule
) -> Tuple[List[float], List[float]]:
    """Resolve the train/eval kernel bandwidths from config and data dimension.

    The median heuristic is unsuitable here, so the bandwidth is instead fixed
    relative to the target dimension ``d``: ``gamma = sqrt(gamma_* * d)``. The
    per-coordinate factors ``gamma_train`` / ``gamma_eval`` are *lists* (default
    ``[1.0]``) and are configured independently, while ``d`` (total target event
    size) is owned by the dataset. Under pixel-wise standardization this keeps
    scores roughly dataset-independent. Passing multiple factors, e.g.
    ``[0.5, 1.0, 2.0]``, turns the kernel score into a multi-bandwidth score (the
    per-bandwidth scores are averaged inside the method); ``[1.0]`` reproduces the
    single-bandwidth score.

    :return: ``(gamma_train, gamma_eval)`` as lists of resolved bandwidth values.
    """
    d = datamodule.get_total_dimension()
    gamma_train = [math.sqrt(f * d) for f in _as_list(cfg.get("gamma_train", [1.0]))]
    gamma_eval = [math.sqrt(f * d) for f in _as_list(cfg.get("gamma_eval", [1.0]))]
    log.info(
        f"Resolved gammas from d={d}: gamma_train={gamma_train}, "
        f"gamma_eval={gamma_eval}"
    )
    return gamma_train, gamma_eval


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # SCORE is fit in two decoupled stages and has its own orchestration (two
    # trainers, stage-2 freezing). Dispatch to it when `staged: true` is set in
    # the config.
    if cfg.get("staged", False):
        return train_staged(cfg)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Output channels are owned by the dataset; they size the backbone head and the
    # predictive method so they stay in sync. Input channels also come from the
    # dataset but are passed straight to the backbone (not expanded by the method).
    data_out_channels: int = datamodule.get_channels()
    data_in_channels: int = datamodule.get_input_channels()

    # Resolve the kernel bandwidths from the per-coordinate scaling factors and the
    # data dimension `d`: gamma = sqrt(gamma_* * d). Scaling by `d` makes the
    # sampling-based scores roughly dataset-independent under pixel-wise
    # standardization. `gamma_train` is used during fit, `gamma_eval` during eval.
    gamma_train, gamma_eval = _resolve_gammas(cfg, datamodule)

    # Per-method params (e.g. loss) come from cfg.methods; out_channels, the
    # event dimension and the resolved bandwidths are derived at runtime and
    # injected as overrides. `event_dim` is the total flattened target size
    # V = C * prod(spatial); only the dense parameterizations need it (Cholesky
    # sizes its head from it), and every other method absorbs it through **kwargs.
    log.info(f"Instantiating predictive method <{cfg.method._target_}>")
    predictive_method: PredictiveMethod = hydra.utils.instantiate(
        cfg.method,
        out_channels=data_out_channels,
        event_dim=datamodule.get_total_dimension(),
        gamma=gamma_train,
        gamma_eval=gamma_eval,
    )

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

    log.info("Assembling LightningPredictiveModel")
    # The `optimizer` bundle nests both the optimizer and its scheduler partials.
    optimizer_cfg = hydra.utils.instantiate(cfg.optimizer.optimizer)
    scheduler_cfg = hydra.utils.instantiate(cfg.optimizer.scheduler)
    model: LightningPredictiveModel = LightningPredictiveModel(
        backbone=backbone,
        predictive_method=predictive_method,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    # Descriptive run name + facet tags instead of a random wandb name.
    _apply_wandb_naming(cfg)
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # LearningRateMonitor requires a logger; drop it when none is configured so a
    # run works with or without W&B (otherwise a `logger=null` run raises in
    # `on_train_start`). `_build_stage_trainer` applies the same filter.
    if not logger:
        callbacks = [c for c in callbacks if not isinstance(c, LearningRateMonitor)]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # Warmup epochs are subtractive: they are spent inside `trainer.max_epochs`
    # (the scheduler carves them off the front of the cosine span), so the
    # configured max_epochs is the total number of epochs run and is passed
    # through untouched.
    _log_warmup_budget(
        _warmup_epochs(scheduler_cfg), cfg.trainer.get("max_epochs"), "Training"
    )
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Training is the default action; evaluation is opt-in via `eval: true`.
    if cfg.get("train", True):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    eval_metrics = {}
    if cfg.get("eval", False):
        log.info("Starting evaluation!")
        # Evaluate the best weights from training with the decoupled protocol.
        ckpt_path = ""
        if trainer.checkpoint_callback is not None:
            ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path:
            log.info(f"Loading best ckpt for evaluation: {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            model.load_state_dict(state)
        else:
            log.warning("Best ckpt not found! Evaluating current weights...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        model.to(device)
        eval_metrics = run_evaluation(
            forward=model,
            predictive_method=predictive_method,
            datamodule=datamodule,
            device=device,
            metrics=cfg.get("metrics"),
            loggers=logger,
            prefix="test",
            metric_kwargs=metric_kwargs_from_cfg(cfg),
        )

    # merge train and evaluation metrics
    metric_dict = {**train_metrics, **eval_metrics}

    return metric_dict, object_dict


def _build_stage_trainer(
    cfg: DictConfig,
    stage: int,
    logger: List[Logger],
    warmup_epochs: int = 0,
) -> Trainer:
    """Build a fresh trainer for one stage of the two-stage fit.

    Each stage gets its own trainer (hence its own callback instances) so the
    early-stopping / checkpoint state does not leak across stages -- the two
    stages monitor different ``val_loss`` quantities (the marginal score vs the
    kernel score of the core). The stage's callbacks come from
    ``cfg.callbacks_stage{1,2}`` (falling back to ``cfg.callbacks``), and the
    per-stage epoch ceiling from ``cfg.stage{1,2}_max_epochs``.

    Each stage is a *separate* ``trainer.fit`` on a fresh trainer, so its epoch
    counter starts at 0 and Lightning stops it once it has run ``max_epochs``
    epochs -- the ceiling is per-stage, not cumulative across the two stages
    (``current_epoch`` lives on the trainer's fit loop, not the shared model, so
    nothing carries over). ``warmup_epochs`` (this stage's linear-warmup epochs)
    is subtractive: it runs *within* the stage's ceiling, so
    ``stage{1,2}_max_epochs`` is the total number of epochs the stage runs and is
    only reported here, not added to.
    """
    callbacks_key = f"callbacks_stage{stage}"
    callbacks_cfg = cfg.get(callbacks_key) or cfg.get("callbacks")
    log.info(f"Instantiating stage-{stage} callbacks from <{callbacks_key}>...")
    callbacks: List[Callback] = instantiate_callbacks(callbacks_cfg)

    # LearningRateMonitor requires a logger; drop it when none is configured so
    # staged runs work with or without W&B.
    if not logger:
        callbacks = [c for c in callbacks if not isinstance(c, LearningRateMonitor)]

    max_epochs = cfg.get(f"stage{stage}_max_epochs")
    overrides = {} if max_epochs is None else {"max_epochs": max_epochs}
    _log_warmup_budget(
        warmup_epochs,
        max_epochs if max_epochs is not None else cfg.trainer.get("max_epochs"),
        f"Stage {stage}",
    )
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, **overrides
    )
    return trainer


def train_staged(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Two-stage fit for :class:`SCORE`.

    Stage 1 fits the mean μ and the marginal variances τ (CRPS by default) to
    convergence; stage 2 freezes them and fits the correlation core to the
    standardized target Z = D⁻¹(Y − μ) with the Gaussian kernel score. Each
    stage is a separate ``trainer.fit`` with its own trainer/callbacks, and the
    stage-1 best weights are loaded before stage 2 so it starts from the converged
    marginals. Evaluation uses the stage-2 best checkpoint (frozen stage-1 μ/τ
    plus the fitted core).

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    data_out_channels: int = datamodule.get_channels()
    data_in_channels: int = datamodule.get_input_channels()

    gamma_train, gamma_eval = _resolve_gammas(cfg, datamodule)

    log.info(f"Instantiating predictive method <{cfg.method._target_}>")
    predictive_method: PredictiveMethod = hydra.utils.instantiate(
        cfg.method,
        out_channels=data_out_channels,
        event_dim=datamodule.get_total_dimension(),
        gamma=gamma_train,
        gamma_eval=gamma_eval,
    )

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
    freeze_backbone = cfg.method.get("freeze_backbone", True)
    if freeze_backbone and not hasattr(backbone, "freeze_for_stage2"):
        raise ValueError(
            "Two-stage (SCORE) training with a frozen backbone requires a "
            "backbone exposing freeze_for_stage2(); every backbone under "
            "configs/backbone does. Pick one of those, or set "
            "method.freeze_backbone=false to finetune instead. Got "
            f"{type(backbone).__name__}."
        )

    log.info("Assembling StagedLightningModel")
    # The `optimizer` bundle nests both the optimizer and its scheduler partials.
    optimizer_cfg = hydra.utils.instantiate(cfg.optimizer.optimizer)
    scheduler_cfg = hydra.utils.instantiate(cfg.optimizer.scheduler)
    # Optional per-stage overrides. Each stage bundle is instantiated only when
    # present; otherwise the model falls back to the shared partials, so a default
    # SCORE run uses the same recipe for both stages. Like the shared bundle,
    # each override nests the optimizer and scheduler under one group.
    def _maybe_partial(key: str, sub: str):
        bundle = cfg.get(key)
        return hydra.utils.instantiate(bundle[sub]) if bundle is not None else None

    scheduler_cfg_stage1 = _maybe_partial("optimizer_stage1", "scheduler")
    scheduler_cfg_stage2 = _maybe_partial("optimizer_stage2", "scheduler")
    model: StagedLightningModel = StagedLightningModel(
        backbone=backbone,
        predictive_method=predictive_method,
        stage1_loss=cfg.method.get("stage1_loss", "crps"),
        stage2_loss=cfg.method.get("stage2_loss", "kernel_score"),
        freeze_backbone=freeze_backbone,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
        optimizer_cfg_stage1=_maybe_partial("optimizer_stage1", "optimizer"),
        scheduler_cfg_stage1=scheduler_cfg_stage1,
        optimizer_cfg_stage2=_maybe_partial("optimizer_stage2", "optimizer"),
        scheduler_cfg_stage2=scheduler_cfg_stage2,
    )

    # Each stage's warmup epochs come from its own scheduler partial, falling back
    # to the shared one when the stage has no override -- mirroring how the model
    # resolves the scheduler in configure_optimizers. They are subtractive: spent
    # inside each stage's trainer max_epochs (see _build_stage_trainer).
    warmup_epochs_stage1 = _warmup_epochs(scheduler_cfg_stage1 or scheduler_cfg)
    warmup_epochs_stage2 = _warmup_epochs(scheduler_cfg_stage2 or scheduler_cfg)

    log.info("Instantiating loggers...")
    _apply_wandb_naming(cfg)
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Build the stage-1 trainer up front so hyperparameter logging (which reads
    # the trainer's loggers) has one available; the stage-2 trainer is built
    # later, after the stage-1 best weights are loaded.
    trainer1 = _build_stage_trainer(
        cfg, stage=1, logger=logger, warmup_epochs=warmup_epochs_stage1
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer1,
    }
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    train_metrics: Dict[str, Any] = {}
    if cfg.get("train", True):
        # --- Stage 1: fit the marginals (mu, tau) to convergence. ---
        # `stage1_ckpt` skips the fit and adopts a previous run's marginals. This
        # is exact: stage 1 optimizes a *marginal* score, and with the
        # renormalization by the implied marginal variances (paper Eq. (10))
        # `get_marg_std` is `diag.sqrt()` independently of the core, so stage 1
        # never sees the core parameters. Runs differing only in the core
        # therefore share the same stage-1 problem, and stage 1 is usually the
        # bulk of the run time.
        #
        # Shape-mismatched entries are dropped before loading: strict=False
        # forgives missing/unexpected *keys* but still raises on a shape
        # mismatch, which is what a core parameterization with a different head
        # width (e.g. filter coefficients instead of one value per mode)
        # produces. Stage 2 retrains the core head from its init anyway, so the
        # donor's core weights are not needed.
        stage1_ckpt = cfg.get("stage1_ckpt")
        if stage1_ckpt:
            log.info(f"Skipping stage 1; adopting marginals from {stage1_ckpt}")
            if not os.path.isfile(stage1_ckpt):
                raise FileNotFoundError(
                    f"stage1_ckpt {stage1_ckpt!r} is not a file. It should point "
                    "at a stage-1 checkpoint from a previous run, e.g. "
                    "logs/train/runs/<date>/checkpoints/stage1/epoch_023.ckpt"
                )
            state = torch.load(stage1_ckpt, map_location="cpu")["state_dict"]
            shapes = {k: v.shape for k, v in model.state_dict().items()}
            dropped = [k for k, v in state.items() if k in shapes and v.shape != shapes[k]]
            if dropped:
                log.info(f"Dropping {len(dropped)} shape-mismatched key(s): {dropped}")
            state = {k: v for k, v in state.items() if k not in dropped}
            missing, unexpected = model.load_state_dict(state, strict=False)
            missing = list(missing) + dropped
            # The mu/diag pathway is what stage 2 freezes and deploys, so a donor
            # that does not supply it would silently evaluate an untrained mean.
            trunk_missing = [k for k in missing if ".other_proj." not in k]
            if trunk_missing:
                raise ValueError(
                    f"stage1_ckpt {stage1_ckpt!r} is missing {len(trunk_missing)} "
                    f"non-core weights, e.g. {trunk_missing[:3]}. It does not "
                    "match this backbone."
                )
            log.info(
                f"Adopted stage-1 weights (core keys left at init: "
                f"{len(missing) - len(trunk_missing)} missing, {len(unexpected)} unused)"
            )
        else:
            log.info("Starting stage-1 training (marginals)!")
            model.set_stage(1)
            trainer1.fit(model=model, datamodule=datamodule)
            if trainer1.checkpoint_callback is not None:
                best1 = trainer1.checkpoint_callback.best_model_path
                if best1:
                    log.info(f"Loading best stage-1 ckpt: {best1}")
                    state = torch.load(best1, map_location="cpu")["state_dict"]
                    model.load_state_dict(state)

        # --- Stage 2: freeze mu/tau, fit the correlation core to convergence. ---
        # Stage 2 gets its own fresh trainer, whose epoch counter starts at 0
        # independently of how many epochs stage 1 ran, so its ceiling is exactly
        # stage2_max_epochs (stage 2's warmup is spent inside that budget).
        log.info("Starting stage-2 training (correlation core)!")
        model.set_stage(2)
        trainer2 = _build_stage_trainer(
            cfg,
            stage=2,
            logger=logger,
            warmup_epochs=warmup_epochs_stage2,
        )
        trainer2.fit(model=model, datamodule=datamodule)
        train_metrics = dict(trainer2.callback_metrics)
        object_dict["trainer"] = trainer2

    eval_metrics: Dict[str, Any] = {}
    if cfg.get("eval", False):
        log.info("Starting evaluation!")
        ckpt_path = ""
        trainer2 = object_dict.get("trainer")
        if trainer2 is not None and trainer2.checkpoint_callback is not None:
            ckpt_path = trainer2.checkpoint_callback.best_model_path
        if ckpt_path:
            log.info(f"Loading best stage-2 ckpt for evaluation: {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            model.load_state_dict(state)
        else:
            log.warning("Best stage-2 ckpt not found! Evaluating current weights...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        model.to(device)
        eval_metrics = run_evaluation(
            forward=model,
            predictive_method=predictive_method,
            datamodule=datamodule,
            device=device,
            metrics=cfg.get("metrics"),
            loggers=logger,
            prefix="test",
            metric_kwargs=metric_kwargs_from_cfg(cfg),
        )

    metric_dict = {**train_metrics, **eval_metrics}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    try:
        # train the model
        metric_dict, _ = train(cfg)
    finally:
        if wandb.run is not None:
            wandb.finish()

    return None


if __name__ == "__main__":
    main()
