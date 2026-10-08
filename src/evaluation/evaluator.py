"""Decoupled evaluation protocol.

:class:`Evaluator` runs a set of registered metrics over a dataloader, given a
predictive method and any callable that maps inputs to processed predictions
(e.g. a trained :class:`~src.models.gaussian_predictive.LightningPredictiveModel`,
which is callable and returns ``process_output(backbone(x))``). It is
independent of the Lightning training loop, so an already-trained checkpoint can
be evaluated standalone.
"""

from typing import Callable, Dict, Iterable, List, Optional

import torch

from src.evaluation.metrics import EVAL_METRICS
from src.methods import PredictiveMethod


class Evaluator:
    """Compute evaluation metrics over a dataloader for a trained model.

    :param predictive_method: the method whose metrics / sampler are used.
    :param metrics: names of metrics to compute (keys of ``EVAL_METRICS``).
        Defaults to all registered metrics.
    :param device: device the forward pass and metrics run on.
    :param metric_kwargs: per-metric extra keyword arguments, keyed by metric
        name (e.g. ``{"variogram_score": {"weight": "inv_dist"}}``). Most metrics
        take none; this is how a configurable one gets its settings without
        every metric growing a parameter list.
    """

    def __init__(
        self,
        predictive_method: PredictiveMethod,
        metrics: Optional[List[str]] = None,
        device: str = "cpu",
        metric_kwargs: Optional[Dict[str, dict]] = None,
    ):
        self.method = predictive_method
        self.metrics = list(metrics) if metrics else list(EVAL_METRICS)
        unknown = [m for m in self.metrics if m not in EVAL_METRICS]
        if unknown:
            raise ValueError(
                f"Unknown metric(s) {unknown}; registered: {list(EVAL_METRICS)}"
            )
        self.device = device
        self.metric_kwargs = dict(metric_kwargs) if metric_kwargs else {}
        unknown_kwargs = [m for m in self.metric_kwargs if m not in EVAL_METRICS]
        if unknown_kwargs:
            raise ValueError(
                f"metric_kwargs names unknown metric(s) {unknown_kwargs}; "
                f"registered: {list(EVAL_METRICS)}"
            )

    @torch.no_grad()
    def evaluate(
        self,
        forward: Callable[[torch.Tensor], torch.Tensor],
        dataloader: Iterable,
    ) -> Dict[str, float]:
        """Run the metrics over ``dataloader`` and return their dataset means.

        :param forward: callable ``x -> pred`` returning processed predictions.
        :param dataloader: yields ``(x, y)`` batches.
        :return: ``{metric_name: mean_value}`` over the whole dataset.
        """
        # Use the larger evaluation sample count for sampling-based scores.
        self.method.training = False

        sums = {name: 0.0 for name in self.metrics}
        total = 0
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = forward(x)
            batch_size = y.shape[0]
            for name in self.metrics:
                kwargs = self.metric_kwargs.get(name, {})
                value = EVAL_METRICS[name](self.method, pred, y, **kwargs)
                # Metrics usually return a scalar Tensor, but some (e.g. the
                # deterministic NLL) return a plain Python float such as
                # ``inf``; coerce both to a number before accumulating.
                value = value.item() if isinstance(value, torch.Tensor) else value
                # Batch-size-weighted so the running mean equals the exact
                # dataset mean even with a ragged final batch.
                sums[name] += value * batch_size
            total += batch_size

        return {name: sums[name] / total for name in self.metrics}
