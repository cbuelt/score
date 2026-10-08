"""Compatibility with runs trained before the methods were renamed to the paper's names.

The predictive methods were renamed to match the paper (SCORE, Diag, LorD, Chol,
SB). Two things still carry the pre-release names and are kept stable on purpose:

* ``_target_`` paths inside the ``config.yaml`` of every trained run.
  :func:`upgrade_legacy_targets` maps them onto the current classes, so old
  checkpoints load unchanged.
* Run names, W&B facet tags and the results-folder slugs (e.g.
  ``phys_lora_r1_kernel_score``). New runs keep producing the same slugs, so
  existing results, tables and notebooks keep matching; :data:`METHOD_SLUGS` maps
  a method config name to the slug it is published under.
"""

from typing import Any, Dict

from omegaconf import DictConfig, ListConfig, open_dict

__all__ = ["LEGACY_TARGETS", "METHOD_SLUGS", "upgrade_legacy_targets"]

#: pre-release ``_target_`` -> current ``_target_``
LEGACY_TARGETS: Dict[str, str] = {
    "src.methods.whitened.WhitenedSpectralDiag": "src.methods.score.SCORE",
    "src.methods.diagonal.PhysDiag": "src.methods.diagonal.Diag",
    "src.methods.lora.PhysLoRa": "src.methods.lord.LorD",
    "src.methods.cholesky.PhysCholesky": "src.methods.cholesky.Cholesky",
    "src.methods.generative.Generative": "src.methods.sample_based.SampleBased",
}

#: method config name (``configs/method/<name>.yaml``) -> slug in run names and tags
METHOD_SLUGS: Dict[str, str] = {
    "score": "whitened_spectral_diag",
    "score_dct": "whitened_dct",
    "score_dct1": "whitened_dct1",
    "score_dct4": "whitened_dct4",
    "score_dst": "whitened_dst",
    "score_uhaar": "whitened_uhaar",
    "score_random": "whitened_random",
    "score_matern": "whitened_matern",
    "diag": "phys_diag",
    "lord": "phys_lora",
    "cholesky": "phys_cholesky",
    "sample_based": "generative",
}


def upgrade_legacy_targets(cfg: Any) -> Any:
    """Rewrite pre-release ``_target_`` paths in ``cfg`` in place and return it."""
    if isinstance(cfg, DictConfig):
        with open_dict(cfg):
            for key in list(cfg.keys()):
                if key == "_target_":
                    target = cfg[key]
                    if isinstance(target, str) and target in LEGACY_TARGETS:
                        cfg[key] = LEGACY_TARGETS[target]
                else:
                    node = cfg._get_node(key)
                    if isinstance(node, (DictConfig, ListConfig)):
                        upgrade_legacy_targets(node)
    elif isinstance(cfg, ListConfig):
        for item in cfg:
            if isinstance(item, (DictConfig, ListConfig)):
                upgrade_legacy_targets(item)
    elif isinstance(cfg, dict):
        for key, value in cfg.items():
            if key == "_target_" and value in LEGACY_TARGETS:
                cfg[key] = LEGACY_TARGETS[value]
            elif isinstance(value, (dict, list)):
                upgrade_legacy_targets(value)
    elif isinstance(cfg, list):
        for item in cfg:
            upgrade_legacy_targets(item)
    return cfg
