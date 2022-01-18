import functools
from typing import Any, Iterable, Optional, Type, cast

import pytorch_yard
from avalanche.benchmarks.scenarios.generic_benchmark_creation import (
    create_multi_dataset_generic_benchmark,
)
from avalanche.benchmarks.scenarios.generic_cl_scenario import GenericScenarioStream
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCExperience
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset
from avalanche.evaluation.metric_definitions import PluginMetric
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins.synaptic_intelligence import SynapticIntelligencePlugin
from pytorch_yard import info, info_bold
from pytorch_yard.avalanche import RichLogger, incremental_domain
from pytorch_yard.avalanche.scenarios import incremental_class, incremental_task
from pytorch_yard.experiments.avalanche import AvalancheExperiment
from rich import print
from torch import Tensor
from torch.optim import SGD, Adam

from intervalnet.cfg import (
    DatasetType,
    OptimizerType,
    ScenarioType,
    Settings,
    StrategyType,
)
from intervalnet.datasets import mnist
from intervalnet.metrics.basic import EvalAccuracy, TotalLoss, TrainAccuracy
from intervalnet.metrics.interval import interval_training_diagnostics
from intervalnet.models.interval import IntervalMLP
from intervalnet.models.mlp import MLP
from intervalnet.strategies import EWCPlugin, JointTraining, LwFPlugin, VanillaTraining
from intervalnet.strategy import IntervalTraining

assert pytorch_yard.__version__ == "2021.12.31.1", "Code not tested with different pytorch-yard versions."  # type: ignore # noqa


class Experiment(AvalancheExperiment):
    def __init__(
        self,
        config_path: str,
        settings_cls: Type[Settings],
        settings_group: Optional[str] = None,
    ) -> None:
        super().__init__(config_path, settings_cls, settings_group=settings_group)

        self.cfg: Settings
        """Experiment config."""

        self.input_size: int
        """Model input size."""

        self.n_output_classes: int
        """Number of classes for each head."""

        self.n_heads: int
        """Number of model heads."""

    def entry(self, root_cfg: pytorch_yard.RootConfig) -> None:
        super().entry(root_cfg)

    def main(self):
        super().main()

        self.setup_dataset()
        self.setup_scenario()

        if self.cfg.strategy is StrategyType.Naive:
            self.setup_naive()
        elif self.cfg.strategy is StrategyType.EWC:
            self.setup_ewc()
        elif self.cfg.strategy is StrategyType.SI:
            self.setup_si()
        elif self.cfg.strategy is StrategyType.LWF:
            self.setup_lwf()
        elif self.cfg.strategy is StrategyType.Interval:
            self.setup_interval()
        elif self.cfg.strategy is StrategyType.Joint:
            self.setup_joint()
        else:
            raise ValueError(f"Unknown strategy type: {self.cfg.strategy}")

        print(self.model)

        self.setup_optimizer()
        self.setup_evaluator()
        self.setup_strategy()

        # ------------------------------------------------------------------------------------------
        # Experiment loop
        # ------------------------------------------------------------------------------------------
        info_bold("Starting experiment...")
        for i, experience in enumerate(cast(Iterable[NCExperience], self.scenario.train_stream)):
            info(f"Start of experience: {experience.current_experience}")
            info(f"Current classes: {experience.classes_in_this_experience}")

            self.strategy.valid_classes = len(experience.classes_seen_so_far)

            if self.cfg.strategy is StrategyType.Joint:
                i = len(self.scenario.train_stream) - 1  # test on all data
                self.strategy.valid_classes = self.scenario.n_classes

            seen_datasets: list[AvalancheDataset[Tensor, int]] = [
                AvalancheDataset(exp.dataset, task_labels=t if self.cfg.scenario is ScenarioType.INC_TASK else 0)  # type: ignore # noqa
                for t, exp in enumerate(self.scenario.test_stream[0 : i + 1])  # type: ignore
            ]
            seen_test = functools.reduce(lambda a, b: a + b, seen_datasets)  # type: ignore
            seen_test_stream: GenericScenarioStream[Any, Any] = create_multi_dataset_generic_benchmark(
                [], [], other_streams_datasets={"seen_test": [seen_test]}
            ).seen_test_stream  # type: ignore

            if self.cfg.strategy is StrategyType.Joint:
                self.strategy.train(self.scenario.train_stream, [self.scenario.test_stream, seen_test_stream])  # type: ignore # noqa
                break  # only one valid experience in joint training
            else:
                self.strategy.train(experience, [self.scenario.test_stream, seen_test_stream])  # type: ignore

            info("Training completed")

    # ------------------------------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------------------------------
    def setup_dataset(self):
        if self.cfg.dataset is DatasetType.MNIST:
            self.train, self.test, self.transforms = mnist()
            self.n_classes = 10
            self.input_size = 28 * 28
        else:
            raise ValueError(f"Unknown dataset type: {self.cfg.dataset}")

    def setup_scenario(self):
        if self.cfg.scenario is ScenarioType.INC_TASK:
            _setup = incremental_task
            self.n_heads = self.cfg.n_experiences
        elif self.cfg.scenario is ScenarioType.INC_DOMAIN:
            _setup = incremental_domain
            self.n_heads = 1
        elif self.cfg.scenario is ScenarioType.INC_CLASS:
            _setup = incremental_class
            self.n_heads = 1
        else:
            raise ValueError(f"Unknown scenario type: {self.cfg.scenario}")

        self.scenario, self.n_classes_per_head = _setup(
            self.train,
            self.test,
            self.transforms,
            self.cfg.n_experiences,
            self.n_classes,
        )

    def setup_optimizer(self):
        if self.cfg.optimizer is OptimizerType.SGD:
            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                momentum=self.cfg.momentum if self.cfg.momentum else 0,
            )
        elif self.cfg.optimizer is OptimizerType.ADAM:
            self.optimizer = Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {self.cfg.optimizer}")

        print(self.optimizer)

    def setup_evaluator(self):
        metrics: list[PluginMetric[Any]] = [
            TotalLoss(),
            TrainAccuracy(),
            EvalAccuracy(),
        ]

        if self.cfg.strategy is StrategyType.Interval:
            assert isinstance(self.model, IntervalMLP)
            metrics += interval_training_diagnostics(self.model)

        self.evaluator = EvaluationPlugin(
            *metrics,
            benchmark=self.scenario,
            loggers=[
                RichLogger(
                    ignore_metrics=[
                        r"Diagnostics/(.*)",
                        r"DiagnosticsHist/(.*)",
                    ]
                ),
                self.wandb_logger,
            ],
        )

    def setup_strategy(self):
        self.strategy = self.strategy_(
            model=self.model,
            optimizer=self.optimizer,
            train_mb_size=self.cfg.batch_size,
            train_epochs=self.cfg.epochs,
            eval_mb_size=self.cfg.batch_size,
            evaluator=self.evaluator,
            device=self.device,
            eval_every=1,
            cfg=self.cfg,
        )
        print(self.strategy)

    # ------------------------------------------------------------------------------------------
    # Experiment variants
    # ------------------------------------------------------------------------------------------
    def _get_mlp_model(self):
        return MLP(
            input_size=self.input_size * 1,
            hidden_dim=400,
            output_classes=self.n_classes_per_head,
            heads=self.n_heads,
        )

    def setup_naive(self):
        self.model = self._get_mlp_model()
        self.strategy_ = functools.partial(
            VanillaTraining,
        )

    def setup_joint(self):
        self.model = self._get_mlp_model()
        self.strategy_ = functools.partial(
            JointTraining,
        )

    def setup_ewc(self):
        self.model = self._get_mlp_model()
        assert self.cfg.ewc_lambda and self.cfg.ewc_mode
        self.strategy_ = functools.partial(
            VanillaTraining,
            plugins=[EWCPlugin(self.cfg.ewc_lambda, self.cfg.ewc_mode, self.cfg.ewc_decay)],
        )

    def setup_si(self):
        self.model = self._get_mlp_model()
        assert self.cfg.si_lambda
        self.strategy_ = functools.partial(
            VanillaTraining,
            plugins=[SynapticIntelligencePlugin(self.cfg.si_lambda)],
        )

    def setup_lwf(self):
        self.model = self._get_mlp_model()
        assert self.cfg.lwf_alpha and self.cfg.lwf_temperature
        self.strategy_ = functools.partial(
            VanillaTraining,
            plugins=[LwFPlugin(self.cfg.lwf_alpha, self.cfg.lwf_temperature)],
        )

    def setup_interval(self):
        self.model = IntervalMLP(
            input_size=self.input_size * 1,
            hidden_dim=400,
            output_classes=self.n_classes_per_head,
            radius_multiplier=self.cfg.interval.radius_multiplier,
            max_radius=self.cfg.interval.max_radius,
            bias=self.cfg.interval.bias,
            heads=self.n_heads,
            normalize_shift=self.cfg.interval.normalize_shift,
            normalize_scale=self.cfg.interval.normalize_scale,
            scale_init=self.cfg.interval.scale_init,
        )
        self.strategy_ = functools.partial(
            IntervalTraining,
            enable_visdom=self.cfg.enable_visdom,
            visdom_reset_every_epoch=self.cfg.visdom_reset_every_epoch,
        )


if __name__ == "__main__":
    Experiment("intervalnet", Settings)
