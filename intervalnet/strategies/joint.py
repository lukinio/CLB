from typing import Any, Optional, Sequence, Union

import torch
import torch.linalg
import torch.nn as nn
from avalanche.benchmarks import Experience
from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from rich import print  # type: ignore # noqa
from torch.optim import Optimizer

from intervalnet.cfg import Settings
from intervalnet.strategies.vanilla import VanillaTraining


class JointTraining(VanillaTraining):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device: torch.device = torch.device("cpu"),
        plugins: Optional[Sequence[StrategyPlugin]] = None,
        evaluator: Optional[EvaluationPlugin] = None,
        eval_every: int = -1,
        *,
        cfg: Settings,
    ):
        super().__init__(  # type: ignore
            model,
            optimizer,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            cfg=cfg,
        )

        # JointTraining can be trained only once.
        self._is_fitted = False

    def train(
        self,
        experiences: Union[Experience, Sequence[Experience]],
        eval_streams: Optional[Sequence[Union[Experience, Sequence[Experience]]]] = None,
        **kwargs: Any,
    ):
        """Repurposed code from Avalanche."""
        self.is_training = True
        self.model.train()
        self.model.to(self.device)

        # Normalize training and eval data.
        if isinstance(experiences, Experience):
            experiences = [experiences]
        if eval_streams is None:
            eval_streams = [experiences]
        for i, exp in enumerate(eval_streams):
            if isinstance(exp, Experience):
                eval_streams[i] = [exp]  # type: ignore

        self._experiences = experiences
        self.before_training(**kwargs)  # type: ignore
        for exp in experiences:
            self.train_exp(exp, eval_streams, **kwargs)  # type: ignore
            # Joint training only needs a single step because
            # it concatenates all the data at once.
            break
        self.after_training(**kwargs)  # type: ignore

        res: dict[Any, Any] = self.evaluator.get_last_metrics()  # type: ignore
        return res

    def train_dataset_adaptation(self, **kwargs: Any):
        """Concatenates all the datastream."""
        self.adapted_dataset = self._experiences[0].dataset  # type: ignore
        for exp in self._experiences[1:]:
            cat_data = AvalancheConcatDataset([self.adapted_dataset, exp.dataset])  # type: ignore
            self.adapted_dataset = cat_data  # type: ignore
        self.adapted_dataset = self.adapted_dataset.train()  # type: ignore
