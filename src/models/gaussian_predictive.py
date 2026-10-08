from functools import partial
from typing import Callable, Optional

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from src.methods import PredictiveMethod
from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR


def _split_weight_decay(params, weight_decay: float):
    """Build two param groups so weight decay skips biases and 1-D (norm) params.

    AdamW applies decoupled weight decay to every group by default; decaying the
    affine/bias parameters of norm layers is the wrong default and hurts. This
    puts 1-D parameters (biases, LayerNorm/BatchNorm weights) in a
    ``weight_decay=0`` group and everything else in the decayed group. ``params``
    is consumed once, so callers pass a list (not a generator).
    """
    decay, no_decay = [], []
    for p in params:
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _build_optimizer_and_scheduler(
    module: "LightningPredictiveModel",
    params: list,
    optimizer_cfg: Optional[Callable] = None,
    scheduler_cfg: Optional[Callable] = None,
):
    """Instantiate the optimizer (with weight-decay-split groups) and the
    warmup+cosine scheduler from the partials, sizing the schedule from the
    attached trainer.

    ``optimizer_cfg`` / ``scheduler_cfg`` default to the module's shared partials;
    :class:`StagedLightningModel` passes its per-stage partials instead. Returns
    the Lightning ``configure_optimizers`` dict. For the two-stage fit each
    stage's fresh ``fit`` re-invokes this with that stage's trainable params and
    ``max_epochs``, giving each stage its own cycle.
    """
    optimizer_cfg = optimizer_cfg or module.optimizer_cfg
    scheduler_cfg = scheduler_cfg or module.scheduler_cfg

    weight_decay = optimizer_cfg.keywords.get("weight_decay", 0.0)
    param_groups = _split_weight_decay(params, weight_decay)
    optimizer = optimizer_cfg(param_groups)

    # Total optimizer steps for this fit; Lightning accounts for max_epochs,
    # devices, and accumulate_grad_batches. This is the whole budget: warmup is
    # subtractive, so trainer.max_epochs is exactly what was configured.
    max_steps = int(module.trainer.estimated_stepping_batches)

    # The schedulers count in optimizer steps, but configs are written in epochs
    # (steps-per-epoch is only knowable here, once the trainer is attached). So
    # every `<name>_epochs` scheduler keyword becomes the `<name>_steps` the
    # scheduler takes: `warmup_epochs` -> `warmup_steps` for all of them, plus
    # e.g. `decay_start_epochs` / `decay_period_epochs` for
    # StepwiseExponentialLR. Warmup is carved off the front of the same budget,
    # so a fit with max_epochs=E and warmup_epochs=W runs W warmup epochs
    # followed by a decay span over the remaining E - W.
    sched_kwargs = dict(scheduler_cfg.keywords)
    max_epochs = module.trainer.max_epochs or 1
    steps_per_epoch = max_steps / max_epochs
    # Every scheduler takes warmup_steps, so supply it even when the partial
    # leaves it out (a scheduler that does not warm up, e.g. type3).
    sched_kwargs.setdefault("warmup_epochs", 0)
    step_kwargs = {}
    for key in [k for k in sched_kwargs if k.endswith("_epochs")]:
        epochs = sched_kwargs.pop(key)
        name = f"{key[: -len('_epochs')]}_steps"
        step_kwargs[name] = min(int(round(epochs * steps_per_epoch)), max_steps)
    scheduler = scheduler_cfg.func(
        optimizer,
        max_steps=max_steps,
        **step_kwargs,
        **sched_kwargs,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }


class LightningPredictiveModel(L.LightningModule):
    """Lightning module for single-stage training of a backbone + predictive method.

    Used for Det, Diag, LorD, Chol and SB; SCORE uses
    :class:`StagedLightningModel`.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        predictive_method: PredictiveMethod,
        optimizer_cfg: Optional[Callable] = None,
        scheduler_cfg: Optional[Callable] = None,
        lr: float = 1e-3,
        track_grad_norms: bool = True,
        **kwargs,
    ):
        super().__init__()
        # Persist init args so W&B can record them as run config later. The
        # backbone (an nn.Module) is already saved via checkpointing, the
        # predictive method is not serializable, and the optimizer/scheduler
        # partials are not loggable, so all are excluded.
        self.save_hyperparameters(
            ignore=[
                "backbone",
                "predictive_method",
                "optimizer_cfg",
                "scheduler_cfg",
            ]
        )
        # Backbone
        self.backbone = backbone
        # Optimizer / scheduler partials (functools.partial of the configured
        # torch optimizer and the warmup+cosine scheduler). They are finalized in
        # configure_optimizers once the param groups and trainer are known. Fall
        # back to sane defaults so the module is usable when built bare (tests).
        self.optimizer_cfg = optimizer_cfg or partial(
            torch.optim.AdamW, lr=lr, weight_decay=0.05
        )
        self.scheduler_cfg = scheduler_cfg or partial(
            LinearWarmupCosineAnnealingLR, warmup_epochs=0, eta_min=1e-6
        )
        # Optimizer learning rate
        self.lr = lr
        # Whether to log gradient-norm diagnostics each optimizer step
        self.track_grad_norms = track_grad_norms
        # Keep the predictive method so its sampling mode can be toggled per step.
        # It is not an nn.Module, so it is not part of the model's state.
        self.predictive_method = predictive_method
        # Process head
        self.process_output = predictive_method.process_output
        # Train loss
        self.train_loss = predictive_method.get_train_loss()
        # Test-set metrics are not computed here: evaluation is handled by the
        # protocol in ``src.evaluation`` (an Evaluator over the metric
        # registry), run as a separate phase after training.

    def forward(self, x, n_samples: Optional[int] = None):
        """Run the backbone and un-stack its head output into ``pred``.

        For the closed-form families the backbone is deterministic and
        ``n_samples`` is ignored: ``pred`` carries distribution parameters and
        the members are drawn from them afterwards, by ``get_samples``.

        For the sample-based method (SB) the members *are* the forward pass
        -- the backbone injects a fresh latent per member -- so the count has to
        be decided here. It defaults to the method's own
        ``n_samples`` (``n_train_samples`` in train/val mode,
        ``n_eval_samples`` outside it), and callers that need a specific ensemble
        size, such as the chunked draw loop in
        :mod:`src.evaluation.calibration`, pass it explicitly.
        """
        if self.predictive_method.samples_from_backbone:
            if n_samples is None:
                n_samples = self.predictive_method.n_samples
            pre_head = self.backbone(x, n_samples=n_samples)
        else:
            pre_head = self.backbone(x)
        output = self.process_output(pre_head)
        return output


    def training_step(self, batch, batch_idx):
        # Use the smaller training sample count for sampling-based scores.
        self.predictive_method.training = True
        x, y = batch
        pred = self(x)
        loss = self.train_loss(pred, y)
        self.log("training/train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation tracks the training loss, so use the training sample count.
        self.predictive_method.training = True
        x, y = batch
        pred = self(x)
        loss = self.train_loss(pred, y)
        self.log("training/val_loss", loss, sync_dist=True)
        return loss

    def predict_step(self, batch):
        self.predictive_method.training = False
        x, y = batch
        pred = self(x)
        return pred

    def configure_optimizers(self):
        params = list(self.backbone.parameters())
        return _build_optimizer_and_scheduler(self, params)

    def _head_module(self):
        """The head whose gradient is most directly driven by the loss.

        Backbone-agnostic. Different backbones name the final head differently:

        * ucast (``DhariwalUNet``), ``PatchTST`` and ``DepthAnything``: a shared
          mu/sigma head ``shared_proj`` plus an optional ``other_proj`` (the
          head of the ``other`` group, e.g. the SCORE core trained in stage 2
          or the LorD factor).
        * ``Multigraph``: a single ``head`` linear for mu/sigma, plus the same
          optional ``other_proj``.

        The stage-2-trained head (``other_proj``) is preferred when present so
        grad logging tracks the actively fitted head. Returns ``None`` if no
        recognised head is present.
        """
        for name in ("other_proj", "head", "shared_proj"):
            head = getattr(self.backbone, name, None)
            if head is not None:
                return head
        return None

    def on_before_optimizer_step(self, optimizer):
        # Log gradient-norm diagnostics. This hook fires after backward() but
        # before the optimizer step (and any gradient clipping), so .grad holds
        # the unmodified gradients. Used to compare gradient behaviour across
        # training losses, e.g. whether NLL gradients explode while the kernel
        # score's stay well-behaved. Norms are L2, matching the convention of
        # Lightning's grad_norm utility and torch's clip_grad_norm_.
        if not self.track_grad_norms:
            return

        total_grads = [
            p.grad.detach().norm(2) for p in self.parameters() if p.grad is not None
        ]
        if not total_grads:
            return
        self.log("grad_norm/total", torch.norm(torch.stack(total_grads), 2),
                 on_step=True, on_epoch=False, prog_bar=False)

        # The final head, whose gradient is most directly driven by the loss.
        head = self._head_module()
        head_grads = (
            [p.grad.detach().norm(2) for p in head.parameters() if p.grad is not None]
            if head is not None
            else []
        )
        if head_grads:
            self.log("grad_norm/head", torch.norm(torch.stack(head_grads), 2),
                     on_step=True, on_epoch=False, prog_bar=False)

    @rank_zero_only
    def on_fit_start(self):
        # Push saved hyperparameters (i.e., model kwargs) to W&B config for this run.
        # Only rank 0 has the real wandb run; other ranks get a DummyExperiment
        # whose `config` is a no-op, so this must run on the main process only.
        if isinstance(self.logger, WandbLogger) and self.logger.experiment is not None:
            params = dict(self.hparams)
            extra_kwargs = params.pop("kwargs", {})
            if isinstance(extra_kwargs, dict):
                params.update(extra_kwargs)
            self.logger.experiment.config.update(params, allow_val_change=True)


class StagedLightningModel(LightningPredictiveModel):
    """Lightning module driving the two-stage fit of SCORE.

    Unlike the base module, which binds a single training loss at init, this
    module is *stage-aware*: :meth:`set_stage` rebinds ``self.train_loss`` to the
    configured per-stage loss and, entering stage 2, freezes the mean/variance
    (μ, τ) pathway via the backbone's ``freeze_for_stage2``. The orchestration in
    :func:`src.train.train.train_staged` calls ``set_stage`` between the two
    ``trainer.fit`` calls.

    * **Stage 1** trains the whole backbone on ``stage1_loss`` (default ``crps``),
      which only touches the mean μ and the marginal variances τ -- the core
      head receives no gradient.
    * **Stage 2** fits the correlation core on ``stage2_loss`` (default
      ``kernel_score``, the Gaussian kernel score of the standardized target
      under the core). With
      ``freeze_backbone=True`` (the default) the trunk and mu/diag heads are
      frozen, so ``configure_optimizers`` -- re-invoked by the fresh stage-2
      ``fit`` -- builds the optimizer over the core head alone and the
      converged stage-1 marginals stay fixed. With ``freeze_backbone=False``
      stage 2 is full finetuning: nothing is frozen and the optimizer covers all
      weights.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        predictive_method: PredictiveMethod,
        stage1_loss: str = "crps",
        stage2_loss: str = "kernel_score",
        freeze_backbone: bool = True,
        optimizer_cfg: Optional[Callable] = None,
        scheduler_cfg: Optional[Callable] = None,
        optimizer_cfg_stage1: Optional[Callable] = None,
        scheduler_cfg_stage1: Optional[Callable] = None,
        optimizer_cfg_stage2: Optional[Callable] = None,
        scheduler_cfg_stage2: Optional[Callable] = None,
        lr: float = 1e-3,
        track_grad_norms: bool = True,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            predictive_method=predictive_method,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            lr=lr,
            track_grad_norms=track_grad_norms,
            **kwargs,
        )
        self.stage1_loss = stage1_loss
        self.stage2_loss = stage2_loss
        self.freeze_backbone = freeze_backbone
        # Optional per-stage optimizer/scheduler partials. Each falls back to the
        # shared optimizer_cfg/scheduler_cfg when unset, so a default SCORE run
        # uses the same recipe for both stages; supplying e.g. optimizer_cfg_stage2
        # lets stage-2 fine-tuning use a smaller lr / shorter warmup independently.
        self.optimizer_cfg_stage1 = optimizer_cfg_stage1
        self.scheduler_cfg_stage1 = scheduler_cfg_stage1
        self.optimizer_cfg_stage2 = optimizer_cfg_stage2
        self.scheduler_cfg_stage2 = scheduler_cfg_stage2
        self.current_stage = 1
        # Start in stage 1; train_loss is set explicitly via set_stage so the two
        # stages cannot disagree with self.loss bound at base init.
        self.set_stage(1)

    def set_stage(self, stage: int) -> None:
        """Switch the active training stage (1 or 2).

        Rebinds ``self.train_loss`` to the configured per-stage loss method on the
        predictive method. Entering stage 2 freezes the shared trunk and the
        mu/diag heads so only the correlation core is trained -- unless
        ``freeze_backbone`` is ``False``, in which case stage 2 is treated as full
        finetuning and all weights stay trainable.
        """
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}.")
        self.current_stage = stage
        loss_name = self.stage1_loss if stage == 1 else self.stage2_loss
        self.train_loss = getattr(self.predictive_method, loss_name)
        if stage == 2 and self.freeze_backbone:
            if not hasattr(self.backbone, "freeze_for_stage2"):
                raise AttributeError(
                    "Stage-2 freezing requires a backbone exposing "
                    "freeze_for_stage2() (every backbone in src/models/backbones "
                    f"does); got {type(self.backbone).__name__}. Set "
                    "method.freeze_backbone=false to finetune instead."
                )
            self.backbone.freeze_for_stage2()

    def configure_optimizers(self):
        # Optimize only the currently-trainable parameters. In stage 2 the trunk
        # and mu/diag heads have requires_grad=False, so this naturally yields an
        # optimizer over the core head alone. Each fit() re-invokes this after
        # set_stage has set the grad flags, so each stage gets its own
        # warmup+cosine cycle sized to that stage's max_epochs.
        params = [p for p in self.backbone.parameters() if p.requires_grad]
        optimizer_cfg, scheduler_cfg = self._stage_optim_partials()
        return _build_optimizer_and_scheduler(
            self, params, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg
        )

    def _stage_optim_partials(self):
        """Pick the optimizer/scheduler partials for the current stage.

        Returns the stage-specific partial when configured, else ``None`` so
        :func:`_build_optimizer_and_scheduler` falls back to the shared
        ``optimizer_cfg`` / ``scheduler_cfg``.
        """
        if self.current_stage == 1:
            return self.optimizer_cfg_stage1, self.scheduler_cfg_stage1
        return self.optimizer_cfg_stage2, self.scheduler_cfg_stage2

    def training_step(self, batch, batch_idx):
        # Log under a stage-suffixed name so the two stages appear as separate
        # curves in W&B (the shared model's global_step carries across the two
        # fit() calls, so a single series would otherwise have stage 2
        # overwrite/continue stage 1's line). The unsuffixed `training/train_loss`
        # is still logged too so each stage's callbacks (checkpoint / early
        # stopping) keep monitoring a stable key, matching the base module's
        # `training/...` convention.
        self.predictive_method.training = True
        x, y = batch
        pred = self(x)
        loss = self.train_loss(pred, y)
        self.log(f"training/train_loss_stage{self.current_stage}", loss, sync_dist=True)
        self.log("training/train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.predictive_method.training = True
        x, y = batch
        pred = self(x)
        loss = self.train_loss(pred, y)
        self.log(f"training/val_loss_stage{self.current_stage}", loss, sync_dist=True)
        self.log("training/val_loss", loss, sync_dist=True)
        return loss
