"""Unpickling shims for checkpoints written by a newer Hydra than this env has.

A Lightning checkpoint carries the run's ``hyper_parameters`` as a pickled
``DictConfig``, so ``torch.load`` has to import every class that config
references -- even when the caller only wants ``state_dict``. Hydra 1.4
defers unresolved ``_target_`` values through
``hydra._internal.target_policy._DeferredTarget``, a class that does not exist
in Hydra 1.3, so a checkpoint trained with the newer release fails to load
under Hydra 1.3 with ``ModuleNotFoundError`` long before any tensor is touched.
Upgrading Hydra is not always an option (e.g. when the environment is pinned to
an older ``torch`` by the GPU driver), so this module provides a shim instead.

Registering a permissive placeholder is safe *for the loaders in this repo*
because none of them read the checkpoint's ``hyper_parameters``:
``calibration.build_model_from_run`` and ``train.py``'s stage handoff both
rebuild the backbone and the method from the run's own ``config.yaml`` and take
only ``state_dict`` from the file. The placeholder keeps whatever state it was
pickled with, so it round-trips rather than silently becoming ``None`` -- but
nothing in this repo inspects it, and a caller that wants a *faithful*
``hyper_parameters`` needs the real Hydra, not this module.
"""

from __future__ import annotations

import sys
import types

__all__ = ["install_hydra_target_policy_shim"]

_MODULE = "hydra._internal.target_policy"


class _DeferredTarget:
    """Placeholder for Hydra 1.4's deferred ``_target_`` wrapper.

    Pickle rebuilds an instance without calling ``__init__``, then hands it the
    pickled state, so accepting anything in both is all that is required.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})

    def __repr__(self):
        return f"{type(self).__name__}({self.__dict__!r})"


def install_hydra_target_policy_shim() -> bool:
    """Make ``hydra._internal.target_policy`` importable if it is not already.

    Returns whether the shim was installed -- ``False`` means the running Hydra
    already provides the module and nothing was touched. Idempotent, so it is
    safe to call from every entry point that loads a checkpoint.
    """
    try:
        __import__(_MODULE)
    except ImportError:
        pass
    else:
        return False
    module = types.ModuleType(_MODULE)
    module.__doc__ = __doc__
    module._DeferredTarget = _DeferredTarget
    sys.modules[_MODULE] = module
    return True
