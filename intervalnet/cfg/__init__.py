from dataclasses import dataclass
from enum import Enum, auto

import pytorch_yard


class DatasetType(Enum):
    MNIST = auto()
    # FASHION_MNIST = auto()
    # CELEBA = auto()
    # CIFAR100 = auto()


class ScenarioType(Enum):
    # INC_TASK = auto()
    INC_DOMAIN = auto()
    # INC_CLASS = auto()


class ModelType(Enum):
    MLP = auto()
    IntervalMLP = auto()


class OptimizerType(Enum):
    SGD = auto()
    ADAM = auto()


# Interval training settings
@dataclass
class IntervalSettings:
    vanilla_loss_threshold: float = 0.03
    expansion_learning_rate: float = 0.01

    radius_multiplier: float = 1.0
    max_radius: float = 1.0
    radius_exponent: float = 0.5

    robust_lambda: float = 0.002

    metric_lookback: int = 10
    robust_accuracy_threshold: float = 0.9


# General experiment settings validation schema & default values
@dataclass
class Settings(pytorch_yard.Settings):
    # ----------------------------------------------------------------------------------------------
    # General experiment settings
    # ----------------------------------------------------------------------------------------------
    seed: int = 1234

    enable_visdom: bool = False
    visdom_reset_every_epoch: bool = False

    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 0.01
    momentum: float = 0.9

    optimizer: OptimizerType = OptimizerType.ADAM

    # ----------------------------------------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------------------------------------
    dataset: DatasetType = DatasetType.MNIST

    # ----------------------------------------------------------------------------------------------
    # Continual learning setup
    # ----------------------------------------------------------------------------------------------
    scenario: ScenarioType = ScenarioType.INC_DOMAIN
    n_experiences: int = 5

    # ----------------------------------------------------------------------------------------------
    # Model settings
    # ----------------------------------------------------------------------------------------------
    model: ModelType = ModelType.MLP

    # ----------------------------------------------------------------------------------------------
    # IntervalNet settings
    # ----------------------------------------------------------------------------------------------
    interval: IntervalSettings = IntervalSettings()
