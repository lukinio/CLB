from avalanche.evaluation.metrics.accuracy import (AccuracyPluginMetric,
                                                   EpochAccuracy)
from avalanche.evaluation.metrics.loss import EpochLoss

from .generic import MetricNamingMixin


class TotalLoss(MetricNamingMixin[float], EpochLoss):
    def __str__(self):
        return "Loss/total"


class TrainAccuracy(MetricNamingMixin[float], EpochAccuracy):
    def __str__(self):
        return "Accuracy"


class EvalAccuracy(MetricNamingMixin[float], AccuracyPluginMetric):
    def __init__(self):
        super().__init__(reset_at='experience', emit_at='experience', mode='eval')  # type: ignore

    def __str__(self):
        return "Accuracy"
