from pathlib import Path
from typing import Generator

import pytest
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

# repo root = three levels up from this file (tests/configs/test_configs.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train",
            return_hydra_config=True,
            # `experiments` is a mandatory group: a run must name one.
            overrides=["experiments=train/time_series/univariate/etth1"],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(PROJECT_ROOT)
            cfg.trainer.max_epochs = 1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.trainer.logger = False
            cfg.datamodule.num_workers = 0
            cfg.datamodule.pin_memory = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> Generator[DictConfig, None, None]:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.datamodule
    assert cfg_train.backbone
    assert cfg_train.trainer
    # The optimizer bundle nests both the optimizer and its scheduler partials.
    assert cfg_train.optimizer.optimizer
    assert cfg_train.optimizer.scheduler
    assert "scheduler" not in cfg_train  # not a top-level group

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.datamodule)
    # in_channels/out_channels are bound at runtime by train.py from the datamodule
    # and the predictive method, so they are not in the backbone config.
    hydra.utils.instantiate(cfg_train.backbone, in_channels=1, out_channels=1)
    hydra.utils.instantiate(cfg_train.trainer)
    assert hydra.utils.instantiate(cfg_train.optimizer.optimizer) is not None
    assert hydra.utils.instantiate(cfg_train.optimizer.scheduler) is not None


def test_variogram_settings_default_to_unweighted(cfg_train: DictConfig) -> None:
    """The shared run settings carry the `variogram_score` knobs, and the default
    is the unweighted score -- `inv_dist` is spatial, so it must not be the
    default for the 1-D time-series experiments."""
    from src.evaluation.evaluate import metric_kwargs_from_cfg

    assert cfg_train.variogram.weight == "none"
    assert cfg_train.variogram.block_size == 256
    assert metric_kwargs_from_cfg(cfg_train) == {
        "variogram_score": {"weight": "none", "block_size": 256}
    }


@pytest.mark.parametrize(
    "experiment, expected",
    [
        # ERA5 is a lat/lon grid, so grid distance *is* physical distance and the
        # inverse-distance weight probes the short-range structure the
        # parameterizations differ in.
        ("train/era5", "inv_dist"),
        ("train/era5_score", "inv_dist"),
        # A depth map is a camera projection: pixel separation is not metric
        # separation (adjacent pixels straddle occlusion boundaries), so there is
        # no distance to weight by.
        ("train/depth", "none"),
        ("train/depth_score", "none"),
    ],
)
def test_2d_experiments_weight_the_variogram_by_their_geometry(
    experiment: str, expected: str
) -> None:
    """The single-channel 2D tasks pick the weight their grid actually supports.

    Both arms of a task must agree, or the baseline and SCORE are
    scored by different metrics and the comparison is meaningless -- which is the
    failure this parametrization is shaped to catch.
    """
    from src.evaluation.evaluate import metric_kwargs_from_cfg
    from src.evaluation.variogram import VARIOGRAM_WEIGHTS

    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name="train", overrides=[f"experiments={experiment}"])

    assert cfg.variogram.weight == expected
    assert cfg.variogram.weight in VARIOGRAM_WEIGHTS
    assert metric_kwargs_from_cfg(cfg)["variogram_score"]["weight"] == expected
    GlobalHydra.instance().clear()


def test_variogram_kwargs_absent_when_block_is_missing() -> None:
    """A config without a `variogram:` block yields no kwargs, so the metric keeps
    its own defaults rather than being handed an empty/partial dict."""
    from omegaconf import OmegaConf

    from src.evaluation.evaluate import metric_kwargs_from_cfg

    assert metric_kwargs_from_cfg(OmegaConf.create({"metrics": None})) == {}
    # Explicit nulls fall through to the metric defaults too.
    partial = OmegaConf.create({"variogram": {"weight": "inv_dist", "block_size": None}})
    assert metric_kwargs_from_cfg(partial) == {
        "variogram_score": {"weight": "inv_dist"}
    }


@pytest.fixture(scope="function")
def cfg_whitened(tmp_path: Path, dinov2_stub_dir: str) -> Generator[DictConfig, None, None]:
    """A two-stage (SCORE) training config.

    The experiment carries the whole model variant -- method, backbone and the
    `staged` flag -- so selecting it is all it takes; nothing is assembled by hand.
    The DepthAnything backbone normally loads its DINOv2 encoder and checkpoint
    from machine-specific paths; the stub hub source (see conftest) and a null
    checkpoint make the config instantiable on any machine.
    """
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train",
            return_hydra_config=True,
            overrides=["experiments=train/depth_score"],
        )
        with open_dict(cfg):
            cfg.paths.root_dir = str(PROJECT_ROOT)
            cfg.paths.output_dir = str(tmp_path)
            cfg.paths.log_dir = str(tmp_path)
            cfg.trainer.max_epochs = 1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.datamodule.num_workers = 0
            cfg.datamodule.pin_memory = False
            cfg.logger = None
            cfg.backbone.dinov2_source_path = dinov2_stub_dir
            cfg.backbone.checkpoint_path = None

    yield cfg

    GlobalHydra.instance().clear()


def test_whitened_config(cfg_whitened: DictConfig) -> None:
    """The two-stage config composes, the SCORE method/backbone instantiate with
    matching channels, and stage-2 freezing leaves only the core head trainable."""
    from src.models.gaussian_predictive import StagedLightningModel

    assert cfg_whitened.staged is True
    assert "callbacks_stage1" in cfg_whitened
    assert "callbacks_stage2" in cfg_whitened

    HydraConfig().set_config(cfg_whitened)

    method = hydra.utils.instantiate(
        cfg_whitened.method, out_channels=2, gamma=10.0, gamma_eval=10.0
    )
    # Mirror train.py: the backbone gets the method's group_widths so it builds
    # the mu/sigma and core ("other") heads; without them there would be no
    # other head and stage-2 freezing could not work.
    backbone = hydra.utils.instantiate(
        cfg_whitened.backbone,
        out_channels=method.out_channels,
        group_widths=method.group_widths,
    )
    assert backbone.out_channels == method.out_channels
    assert hasattr(backbone, "freeze_for_stage2")
    # The core spectrum is constrained to unit mean by default (the scale
    # gauge), so the predictive marginal variance is exactly the stage-1 tau.
    assert method.normalize_core is True

    model = StagedLightningModel(
        backbone=backbone,
        predictive_method=method,
        stage1_loss=cfg_whitened.method.stage1_loss,
        stage2_loss=cfg_whitened.method.stage2_loss,
    )
    assert model.train_loss.__name__ == cfg_whitened.method.stage1_loss

    model.set_stage(2)
    assert model.train_loss.__name__ == cfg_whitened.method.stage2_loss
    # On DepthAnything the correlation core lives in the "other" head of the DPT
    # decoder; stage 2 must leave exactly that (block + projection) trainable.
    trainable = {n for n, p in backbone.named_parameters() if p.requires_grad}
    assert trainable
    assert all(
        n.startswith(("depth_head.other_block.", "depth_head.other_proj."))
        for n in trainable
    ), trainable


@pytest.mark.parametrize("method", ["score_dct", "score_dst", "score_uhaar"])
def test_analytic_transform_configs_are_dataset_agnostic(method: str) -> None:
    """The DCT/DST/unbalanced-Haar configs must not bake in an event shape.

    A time-series target has a *single* axis (the forecast horizon) while a
    depth target is a 2-D grid, so the shape must come from the data and the
    same config has to work on both ranks. The unbalanced-Haar basis is what
    makes a wavelet domain possible here at all: the classical `haar` kind
    needs a power-of-two axis, which no forecast horizon here is.
    """
    import torch

    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiments=train/time_series/univariate/etth1_score",
                f"method={method}",
            ],
        )
    assert cfg.method.transform.shape is None

    predictive = hydra.utils.instantiate(
        cfg.method, out_channels=1, gamma=10.0, gamma_eval=10.0
    )
    # Non-flat basis -> the renormalization by the implied marginal variances
    # Delta(Sigma_hat) is on, which is what actually exercises the transform.
    assert predictive._use_delta

    pred_len = cfg.datamodule.pred_len
    head = torch.randn(2, 3, pred_len)
    pred = predictive.process_output(head)
    y = torch.randn(2, 1, pred_len)
    assert torch.isfinite(predictive.kernel_score(pred, y))
    assert predictive.transform.spatial_shape == (pred_len,)
    GlobalHydra.instance().clear()

def _require_data(experiment: str) -> None:
    """Skip when the experiment's dataset is not available locally."""
    from src.utils.paths import configured_path

    if "euppbench" in experiment:
        path = configured_path("euppbench_dir") / "meta.npz"
    else:
        path = configured_path("time_series_dir") / "ETT-small" / "ETTh1.csv"
    if not path.exists():
        pytest.skip(f"{path} not available")


@pytest.mark.realdata
@pytest.mark.parametrize(
    "experiment, expected_dim",
    [
        ("train/euppbench", 115),
        ("train/time_series/univariate/etth1", 96),
    ],
)
def test_phys_cholesky_composes_and_sizes_from_the_data(
    experiment: str, expected_dim: int
) -> None:
    """The dense Chol method composes on both low-dimensional tasks.

    ``event_dim`` is the one thing this method cannot infer later -- it sizes the
    head, so it has to come from the datamodule at instantiation, exactly as
    train.py injects it. Pin both that the datamodule reports the dimension this
    task actually has and that the head width follows from it.
    """
    _require_data(experiment)
    import torch

    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train",
            overrides=[f"experiments={experiment}", "method=cholesky"],
        )

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    event_dim = datamodule.get_total_dimension()
    assert event_dim == expected_dim

    predictive = hydra.utils.instantiate(
        cfg.method,
        out_channels=datamodule.get_channels(),
        event_dim=event_dim,
        gamma=10.0,
        gamma_eval=10.0,
    )
    channels = datamodule.get_channels()
    assert predictive.group_widths == {
        "mu": channels,
        "sigma": channels,
        "other": (event_dim - 1) * channels,
    }

    torch.manual_seed(0)
    head = torch.randn(2, predictive.out_channels, event_dim // channels)
    pred = predictive.process_output(head)
    y = torch.randn(2, channels, event_dim // channels)
    assert torch.isfinite(predictive.nll(pred, y))
    assert torch.isfinite(predictive.kernel_score(pred, y))
    GlobalHydra.instance().clear()


@pytest.mark.realdata
@pytest.mark.parametrize(
    "experiment",
    [
        "train/euppbench",
        "train/time_series/univariate/etth1",
        "train/time_series/multivariate/etth1",
    ],
)
def test_generative_composes_and_builds_a_sampling_backbone(experiment: str) -> None:
    """The SB (sample-based generative) arm composes on every backbone and wires its noise through.

    The chain this pins is the one thing no unit test sees in one piece: the
    method's ``noise`` spec has to reach the backbone constructor (train.py
    passes it beside ``group_widths``), the backbone has to build an injector
    from it, and the head output has to un-stack into ``M`` members of the
    dataset's own event shape.
    """
    _require_data(experiment)
    import torch

    from src.models.gaussian_predictive import LightningPredictiveModel

    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train",
            overrides=[f"experiments={experiment}", "method=sample_based"],
        )

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    channels = datamodule.get_channels()
    predictive = hydra.utils.instantiate(
        cfg.method,
        out_channels=channels,
        event_dim=datamodule.get_total_dimension(),
        gamma=10.0,
        gamma_eval=10.0,
    )
    # The shipped default: the fair ensemble Gaussian kernel score, so this arm
    # is fitted on the same functional the tables report for every other method.
    assert predictive.loss == "kernel_score"
    assert predictive.samples_from_backbone is True
    # The head is one group wide however many members are drawn.
    assert predictive.out_channels == channels

    backbone = hydra.utils.instantiate(
        cfg.backbone,
        in_channels=datamodule.get_input_channels(),
        out_channels=predictive.out_channels,
        group_widths=predictive.group_widths,
        group_denorm=predictive.group_denorm,
        other_zero_init=predictive.other_zero_init,
        noise=predictive.noise,
    )
    assert backbone.noise_injector is not None

    model = LightningPredictiveModel(backbone=backbone, predictive_method=predictive)
    model.eval()
    datamodule.setup("fit")
    x, y = next(iter(datamodule.train_dataloader()))
    x, y = x[:2], y[:2]
    predictive.training = True
    with torch.no_grad():
        pred = model(x)
    assert pred.shape == (2, *y.shape[1:], predictive.n_train_samples)
    assert torch.isfinite(predictive.get_train_loss()(pred, y))
    GlobalHydra.instance().clear()


@pytest.mark.parametrize(
    "experiment",
    [
        "train/time_series/univariate/etth1",
        "train/time_series/univariate/etth1_score",
        "train/euppbench",
        "train/euppbench_score",
        "train/era5",
        "train/depth",
    ],
)
def test_benchmark_config_composes_on_every_experiment(experiment: str) -> None:
    """``configs/benchmark.yaml`` accepts the same experiment tree as training.

    The runtime benchmark reuses the training experiment configs verbatim so its
    numbers describe the models that are actually trained. That only holds while
    its defaults list carries every group those configs override -- a missing
    `trainer` or `callbacks` entry would make a subset of experiments fail to
    compose, and the arm would quietly drop out of the cost table.
    """
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="benchmark",
            return_hydra_config=True,
            overrides=[f"experiments={experiment}"],
        )

    # The experiment tree owns `task_name` (it is composed last and sets
    # "train"), so the benchmark pins its own output directory instead -- this is
    # what keeps a cost run out of the training log tree.
    assert "/benchmark/" in cfg.hydra.run.dir
    assert cfg.benchmark.measure_steps > 0
    assert cfg.benchmark.warmup_steps > 0
    assert len(cfg.benchmark.seeds) >= 1
    # The staged flag drives whether both stages are timed.
    assert isinstance(cfg.get("staged"), bool)
    GlobalHydra.instance().clear()


def test_benchmark_pins_one_batch_size_for_every_arm() -> None:
    """``benchmark.batch_size`` overrides the dataset's own value.

    A per-step time is only comparable across methods at a fixed batch size, and
    the datasets carry different defaults, so the pin is what makes the table
    mean anything. ``None`` deliberately falls back to the dataset value, for the
    single-dataset case where nothing is memory-bound.
    """
    from src.train.benchmark import _pin_batch_size

    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="benchmark", overrides=["experiments=train/euppbench"]
        )

    assert _pin_batch_size(cfg, None) == cfg.datamodule.batch_size
    assert _pin_batch_size(cfg, 4) == 4
    assert cfg.datamodule.batch_size == 4
    # Kept in lockstep so a datamodule sizing buffers from the eval loader does
    # not allocate for a shape the benchmark never runs.
    assert cfg.datamodule.eval_batch_size == 4
    GlobalHydra.instance().clear()


def test_benchmark_staged_weighting_uses_the_configured_stage_budgets() -> None:
    """The two-stage headline cost is the epoch-weighted mean of the two stage costs.

    Stage 2 is both cheaper (frozen trunk) and shorter than stage 1, so quoting
    either stage alone misstates what the method costs end to end. This pins the
    weights to the budgets a real run spends, not to a 50/50 split.
    """
    from src.train.benchmark import _weighted_effective

    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="benchmark",
            overrides=["experiments=train/euppbench_score"],
        )

    rows = [
        {"stage": 1, "step_ms_median": 100.0},
        {"stage": 2, "step_ms_median": 50.0},
    ]
    # euppbench_score: 30 stage-1 epochs, 10 stage-2 epochs.
    assert cfg.stage1_max_epochs == 30 and cfg.stage2_max_epochs == 10
    assert _weighted_effective(rows, cfg) == pytest.approx(87.5)
    # A single-stage run has nothing to weight.
    assert _weighted_effective([{"stage": None, "step_ms_median": 1.0}], cfg) is None
    GlobalHydra.instance().clear()
