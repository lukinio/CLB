import functools
from typing import Any, Iterable, Optional, Type, cast

import torch
import torch.nn as nn
import torch.utils.data
from avalanche.benchmarks.scenarios.generic_benchmark_creation import \
    create_multi_dataset_generic_benchmark
from avalanche.benchmarks.scenarios.generic_cl_scenario import \
    GenericScenarioStream
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import (
    NCExperience, NCScenario)
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset
from avalanche.logging import InteractiveLogger
from avalanche.training import Naive
from avalanche.training.plugins.evaluation import EvaluationPlugin
from torch import Tensor
from torch.optim import SGD

import pytorch_yard
from intervalnet.cfg import DatasetType, ModelType, Settings
from intervalnet.datasets import mnist
from intervalnet.metrics.basic import EvalAccuracy, TotalLoss, TrainAccuracy
from intervalnet.metrics.interval import RobustAccuracy, radius_diagnostics
from intervalnet.models.mlp import MLP, IntervalMLP
from intervalnet.strategy import IntervalTraining
from pytorch_yard import info, info_bold
from pytorch_yard.avalanche import RichLogger, incremental_domain


class Experiment(pytorch_yard.Experiment):

    def __init__(self, config_path: str, settings_cls: Type[Settings], settings_group: Optional[str] = None) -> None:
        super().__init__(config_path, settings_cls, settings_group=settings_group, experiment_variant='avalanche')

        self.cfg: Settings
        """ Experiment config. """

        self.train: torch.utils.data.Dataset[torch.Tensor]
        """ Train dataset. """

        self.test: torch.utils.data.Dataset[torch.Tensor]
        """ Test dataset. """

        self.transforms: Any
        """ Transforms applied to train and test data. """

        self.n_classes: int
        """ Number of classes in the dataset. """

        self.input_size: int
        """ Model input size. """

        self.n_output_classes: int
        """ Number of distinct classes for evaluation purposes. """

        self.scenario: NCScenario
        """ Main scenario object. """

        self.model: nn.Module
        """ Main PyTorch model. """

    def main(self, root_cfg: pytorch_yard.RootConfig):
        super().main(root_cfg)

        # ------------------------------------------------------------------------------------------
        # Init
        # ------------------------------------------------------------------------------------------
        self.cfg = cast(Settings, root_cfg.cfg)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.setup_dataset()
        self.setup_scenario()

        # ------------------------------------------------------------------------------------------
        # Experiment variants
        # ------------------------------------------------------------------------------------------
        assert self.cfg.model in ModelType
        if self.cfg.model == ModelType.MLP:
            model_ = MLP
            strategy_ = functools.partial(
                Naive,
                criterion=nn.CrossEntropyLoss(),
            )
        else:
            assert self.cfg.model == ModelType.IntervalMLP
            model_ = IntervalMLP
            strategy_ = IntervalTraining

        # ------------------------------------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------------------------------------
        # Model
        self.model = model_(
            input_size=28 * 28 * 1,
            hidden_dim=400,
            output_classes=self.n_output_classes,
        )
        print(self.model)

        # Evaluation plugin
        metrics = [
            TotalLoss(),
            TrainAccuracy(),
            EvalAccuracy(),
        ]

        if self.cfg.model == ModelType.IntervalMLP:
            metrics.append(RobustAccuracy())
            metrics += radius_diagnostics(self.model)

        eval_plugin = EvaluationPlugin(
            *metrics,
            benchmark=self.scenario,
            loggers=[
                RichLogger(ignore_metrics=[
                    r'Diagnostics/(.*)',
                    r'DiagnosticsHist/(.*)',
                ]),
                self.wandb_logger
            ]
        )

        # Strategy
        strategy = strategy_(
            model=self.model,
            optimizer=SGD(self.model.parameters(), lr=self.cfg.learning_rate),
            train_mb_size=self.cfg.batch_size,
            train_epochs=self.cfg.epochs,
            eval_mb_size=self.cfg.batch_size,
            evaluator=eval_plugin,
            device=self.device,
            eval_every=1,
        )
        print(strategy)

        # ------------------------------------------------------------------------------------------
        # Experiment loop
        # ------------------------------------------------------------------------------------------
        info_bold('Starting experiment...')
        for i, experience in enumerate(cast(Iterable[NCExperience], self.scenario.train_stream)):
            info(f'Start of experience: {experience.current_experience}')
            info(f'Current classes: {experience.classes_in_this_experience}')

            seen_datasets: list[AvalancheDataset[Tensor, int]] = [
                exp.dataset for exp in self.scenario.test_stream[0:i+1]  # type: ignore
            ]
            seen_test = functools.reduce(lambda a, b: a + b, seen_datasets)  # type: ignore
            seen_test_stream: GenericScenarioStream[Any, Any] = create_multi_dataset_generic_benchmark(
                [], [], other_streams_datasets={'seen_test': [seen_test]}
            ).seen_test_stream  # type: ignore

            strategy.train(experience, [self.scenario.test_stream, seen_test_stream])  # type: ignore
            info('Training completed')

    def setup_dataset(self):
        assert self.cfg.dataset in DatasetType
        if self.cfg.dataset == DatasetType.MNIST:
            self.train, self.test, self.transforms = mnist()
            self.n_classes = 10
            self.input_size = 28 * 28

    def setup_scenario(self):
        self.scenario, self.n_output_classes = incremental_domain(
            self.train,
            self.test,
            self.transforms,
            self.cfg.n_experiences,
            self.n_classes
        )


if __name__ == '__main__':
    Experiment('intervalnet', Settings)
