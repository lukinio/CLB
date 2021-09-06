from typing import Optional, Sequence

import torch.nn as nn
from avalanche.training import BaseStrategy
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from torch import Tensor
from torch.optim import Optimizer

# class VAELoss(Module):
#     """
#     VAE optimization criterion.
#     """

#     def __init__(self, kld_weight: float = 1, disable_gaussian_latent: bool = False):
#         super().__init__()

#         self.kld_weight = kld_weight
#         self.disable_gaussian_latent = disable_gaussian_latent

#         self.rec_loss: Tensor
#         self.kld_loss: Tensor

#     def forward(self, output: VAEOutput, target: Tensor):  # type: ignore
#         batch_size = target.shape[0]

#         rec_x, (mu, log_var, _) = output

#         self.rec_loss = F.binary_cross_entropy(rec_x, target, reduction='sum') / batch_size
#         self.kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp()) / batch_size

#         if not self.disable_gaussian_latent:
#             return self.rec_loss + self.kld_weight * self.kld_loss
#         else:
#             return self.rec_loss


class IntervalTraining(BaseStrategy):
    def __init__(self, model: nn.Module, optimizer: Optimizer, train_mb_size: int = 1,
                 train_epochs: int = 1, eval_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence[StrategyPlugin]] = None,
                 evaluator: Optional[EvaluationPlugin] = None, eval_every: int = -1):

        self.mb_output_all: Tensor
        """ All model's outputs computed on the current mini-batch (lower, middle, upper bounds). """

        criterion = nn.CrossEntropyLoss()

        super().__init__(model, optimizer, criterion=criterion, train_mb_size=train_mb_size,
                         train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device,
                         plugins=plugins, evaluator=evaluator, eval_every=eval_every)

    def after_forward(self, **kwargs):
        self.mb_output_all = self.mb_output
        self.mb_output = self.mb_output[:, 1, :].rename(None)  # middle bound

        super().after_forward(**kwargs)

    def after_eval_forward(self, **kwargs):
        self.mb_output_all = self.mb_output
        self.mb_output = self.mb_output[:, 1, :].rename(None)  # middle bound

        super().after_eval_forward(**kwargs)

    # def criterion(self):
    #     middle_output = self.mb_output[:, 0, :].rename(None)
    #     return self._criterion(middle_output, self.mb_y)
