from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

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


# Experiment settings validation schema & default values
@dataclass
class Settings(pytorch_yard.Settings):
    # ----------------------------------------------------------------------------------------------
    # General experiment settings
    # ----------------------------------------------------------------------------------------------
    seed: int = 1234

    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 0.01
    momentum: float = 0.9

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
    vanilla_loss_threshold: Optional[float] = None
    robust_loss_threshold: Optional[float] = None
    radius_multiplier: Optional[float] = None

    l1_lambda: Optional[float] = None
    metric_lookback: Optional[int] = None

    enable_visdom: bool = False
    visdom_reset_every_epoch: bool = False
