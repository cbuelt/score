"""Data and results locations outside of a Hydra run (notebooks, scripts, tests).

Reads ``configs/paths/default.yaml`` merged with the optional, git-ignored
``configs/local/default.yaml``, i.e. the same values a training run resolves.
"""

from pathlib import Path

from omegaconf import OmegaConf

__all__ = ["PROJECT_ROOT", "configured_path"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configured_path(key: str) -> Path:
    """The configured location ``paths.<key>``, e.g. ``"time_series_dir"``."""
    cfg = OmegaConf.create({"paths": OmegaConf.load(PROJECT_ROOT / "configs/paths/default.yaml")})
    local = PROJECT_ROOT / "configs/local/default.yaml"
    if local.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(local))
    cfg.paths.root_dir = str(PROJECT_ROOT)
    value = OmegaConf.select(cfg, f"paths.{key}", throw_on_missing=True)
    if value is None:
        raise KeyError(f"paths.{key} is not configured")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
