from typing import Any, Optional, Sequence, cast

import torch
import torch.linalg
import torch.nn as nn
import torch.nn.functional as F
import visdom
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.training import BaseStrategy
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from rich import print  # type: ignore
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torch.optim import Optimizer
from torchmetrics.functional.classification.accuracy import accuracy as _acc

from intervalnet.models.interval import IntervalMLP, Mode



class IntervalTraining(BaseStrategy):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device: torch.device = torch.device('cpu'),
        plugins: Optional[Sequence[StrategyPlugin]] = None,
        evaluator: Optional[EvaluationPlugin] = None,
        eval_every: int = -1,
        enable_visdom: bool = False,
        *,
        regression_task: bool,
        vanilla_loss_threshold: float,
        robust_loss_threshold: float,
        radius_multiplier: float,
        l1_lambda: float,
        optimizer_cls,
        learning_rate: float,
        eps: float,
        fisher_mode: str,
    ):

        self.mb_output_all: dict[str, Tensor]
        """ All model's outputs computed on the current mini-batch (lower, middle, upper bounds) per layer. """

        self.mb_it: int
        self.training_exp_counter: int

        self.device: torch.device
        self.model: IntervalMLP
        self.regression_task = regression_task
        self.eps = eps

        self.vanilla_loss: Optional[Tensor] = None
        self.robust_loss: Optional[Tensor] = None
        if not self.regression_task:
            self.accuracy_meter = Accuracy()
            self.robust_accuracy_meter = Accuracy()
        self.l1_penalty: Optional[Tensor] = None
        self.robust_penalty: Optional[Tensor] = None
        self.bounds_penalty: Optional[Tensor] = None
        self.radius_penalty: Optional[Tensor] = None
        self.radius_mean: Optional[Tensor] = None

        self.radius_mean_per_layer: dict[str, Optional[Tensor]] = {}
        self.bounds_width_per_layer: dict[str, Optional[Tensor]] = {}

        assert vanilla_loss_threshold is not None
        assert robust_loss_threshold is not None
        assert radius_multiplier is not None
        assert l1_lambda is not None

        self.vanilla_loss_threshold = torch.tensor(vanilla_loss_threshold)
        self.robust_loss_threshold = torch.tensor(robust_loss_threshold)
        self.radius_multiplier = torch.tensor(radius_multiplier)
        self.l1_lambda = torch.tensor(l1_lambda)
        self.optimizer_cls = optimizer_cls
        self.learning_rate = learning_rate
        self.fisher_mode = fisher_mode

        if regression_task:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        super().__init__(model, optimizer, criterion=criterion, train_mb_size=train_mb_size,  # type: ignore
                         train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device,
                         plugins=plugins, evaluator=evaluator, eval_every=eval_every)

        self.model.set_radius_multiplier(self.radius_multiplier)

        self.viz = visdom.Visdom() if enable_visdom else None
        self.viz_debug = visdom.Visdom(env='debug') if enable_visdom else None
        self.windows: dict[str, str] = {}

    @property
    def mode(self):
        return self.model.mode

    @property
    def mode_numeric(self) -> Tensor:
        return torch.tensor(self.mode.value).float()

    def after_forward(self, **kwargs: Any):
        self.mb_output_all = self.mb_output  # type: ignore
        self.mb_output = self.mb_output['last'][:, 1, :].rename(None)  # type: ignore  # middle bound

        super().after_forward(**kwargs)  # type: ignore

    def after_eval_forward(self, **kwargs: Any):
        self.mb_output_all = self.mb_output  # type: ignore
        self.mb_output = self.mb_output['last'][:, 1, :].rename(None)  # type: ignore  # middle bound

        super().after_eval_forward(**kwargs)  # type: ignore

    def bounds_width(self, layer_name: str):
        bounds: Tensor = self.mb_output_all[layer_name].rename(None)  # type: ignore
        return bounds[:, 2, :] - bounds[:, 0, :]

    def apply_fisher_scaling(self, inv_fisher, scale, first_time):
        with torch.no_grad():
            for n, m in self.model.named_interval_children():
                if first_time:
                    m.radius[:, :-1] = inv_fisher[n + "." + "weight"] * scale
                    m.radius[:, -1] = inv_fisher[n + "." + "bias"] * scale
                else:
                    if True:
                        prev_weight_lower = m.prev_weight - m.radius[:, :-1]
                        prev_weight_upper = m.prev_weight + m.radius[:, :-1]
                        assert (prev_weight_lower <= m.weight).all()
                        assert (prev_weight_upper >= m.weight).all()
                        max_weight_radius = torch.minimum(m.weight - prev_weight_lower,
                                                          prev_weight_upper - m.weight)
                        m.radius[:, :-1] = torch.minimum(max_weight_radius,
                                                         inv_fisher[n + "." + "weight"] * scale)

                        prev_bias_lower = m.prev_bias - m.radius[:, -1]
                        prev_bias_upper = m.prev_bias + m.radius[:, -1]
                        assert (prev_bias_lower <= m.bias).all()
                        assert (prev_bias_upper >= m.bias).all()
                        max_bias_radius = torch.minimum(m.bias - prev_bias_lower,
                                                        prev_bias_upper - m.bias)
                        m.radius[:, -1] = torch.minimum(max_bias_radius,
                                                        inv_fisher[n + "." + "bias"] * scale)
                                                   
                    else:
                        m.radius[:, :-1] = torch.minimum(m.radius[:, :-1],
                                                         inv_fisher[n + "." + "weight"] * scale)
                        m.radius[:, -1] = torch.minimum(m.radius[:, -1],
                                                        inv_fisher[n + "." + "bias"] * scale)

    def before_backward(self, **kwargs: Any):
        super().before_backward(**kwargs)  # type: ignore

        # Save base loss for reporting
        self.loss = cast(Tensor, self.loss)  # type: ignore  # Fix Avalanche type-checking
        self.vanilla_loss = self.loss.clone().detach()  # type: ignore

        self.robust_loss = cast(Tensor, self._criterion(self.robust_output(), self.mb_y))  # type: ignore

        # Additional penalties
        self.l1_penalty = torch.tensor(0.0, device=self.device)
        self.robust_penalty = torch.tensor(0.0, device=self.device)
        self.radius_penalty = torch.tensor(0.0, device=self.device)
        self.bounds_penalty = torch.tensor(0.0, device=self.device)

        if not self.regression_task:
            acc = _acc(self.mb_output, self.mb_y)  # type: ignore
            robust_acc = _acc(self.robust_output(), self.mb_y)  # type: ignore

        if self.mode == Mode.EXPANSION:
            # ---------------------------------------------------------------------------------------------------------
            # Expansion phase
            # ---------------------------------------------------------------------------------------------------------
            # === Robust loss ===
            # Maintain an acceptable increase in worst-case loss

            # self.robust_penalty = self.robust_loss * F.relu(acc - robust_acc - 0.10) * 100 / 2

            robust_overflow = F.relu(self.robust_loss - self.robust_loss_threshold) / self.robust_loss_threshold
            # Force quasi-hard constraint
            self.robust_penalty = ((robust_overflow + 1).pow(2) - 1).sqrt()

            # === Radius penalty ===
            # Maximize interval size up to radii of 1
            radii: list[Tensor] = []
            for name, module in self.model.named_interval_children():
                radii.append(module.radius.flatten())

            # INFO: sqrt -> more sparse, push to saturate
            # INFO: pow(2) -> more regular, push to smooth distribution
            # :: Flat version:
            # self.radius_penalty = F.relu(torch.tensor(1.0) - torch.cat(radii)).pow(2).mean()
            # :: Per layer version:
            self.radius_penalty = torch.stack([F.relu(self.radius_multiplier - r).pow(2).mean() for r in radii]).mean()

            # === Bounds penalty ===
            # bounds = [self.bounds_width(name).flatten() for name, _ in self.model.named_children()]
            # self.bounds_penalty = torch.cat(bounds).pow(2).mean().sqrt()

        elif self.mode == Mode.CONTRACTION:
            # ---------------------------------------------------------------------------------------------------------
            # Contraction phase
            # ---------------------------------------------------------------------------------------------------------
            # === Robust loss ===
            #
            pass

            # === Radius (contraction) penalty ===
            pass

        # weights = torch.cat([m.weight.flatten() for m in self.model.interval_children()])
        # l1: Tensor = torch.linalg.vector_norm(weights, ord=1) / weights.shape[0]  # type: ignore
        # self.l1_penalty += self.l1_lambda * l1

        self.loss += self.l1_penalty
        self.loss += self.robust_penalty
        self.loss += self.radius_penalty
        # self.loss += self.bounds_penalty

        # ---------------------------------------------------------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------------------------------------------------------
        radii = []

        for name, module in self.model.named_interval_children():
            radii.append(module.radius.detach().cpu().flatten())
            self.radius_mean_per_layer[name] = radii[-1].mean()
            self.bounds_width_per_layer[name] = self.bounds_width(name).mean()

            if self.viz and self.mb_it == len(self.dataloader) - 1:  # type: ignore
                self.windows[f'{name}.radius'] = self.viz.heatmap(
                    module.radius,
                    win=self.windows.get(f'{name}.radius'),
                    opts={'title': f'{name}.radius --> epoch {(self.epoch or 0) + 1}'}
                )

                self.windows[f'{name}.weight'] = self.viz.heatmap(
                    module.weight.abs().clamp(max=module.weight.abs().quantile(0.99)),
                    win=self.windows.get(f'{name}.weight'),
                    opts={'title': f'{name}.weight.abs() (w/o outliers) --> epoch {(self.epoch or 0) + 1}'}
                )

        if self.viz_debug:
            self.viz_debug.line(X=torch.tensor([self.mb_it]), Y=torch.tensor([acc]),
                                win=self.windows['accuracy'], update='append', name='acc')
            self.viz_debug.line(X=torch.tensor([self.mb_it]), Y=torch.tensor([robust_acc]),
                                win=self.windows['accuracy'], update='append', name='robust_acc')

        self.radius_mean = torch.cat(radii).mean()

    def compute_fisher(self):
        params = {n: p for n, p in self.model.named_parameters() if "weight" in n or "bias" in n}  # TODO: .weight, .bias
        if self.fisher_mode == "identity":
            return None, {n: torch.ones_like(p) for n, p in params.items()}

        fisher = {}
        for n, p in params.items():
            fisher[n] = p.clone().detach().fill_(0)

        self.model.eval()

        train_loader = DataLoader(Subset(self.experience.dataset, list(range(10000))), batch_size=1)
        for i, (input, target, _) in enumerate(train_loader):
            input = input.cuda()

            preds = self.model(input)
            _, m_pred, _ = preds['last'].unbind('bounds')
            m_pred = m_pred.rename(None)


            if self.regression_task:
                loss = m_pred
            else:
                ind = m_pred.max(1)[1].flatten()
                loss = self._criterion(m_pred, ind)

            # empirical fisher?
            self.model.zero_grad()
            loss.backward()

            for n, p in fisher.items():
                if params[n].grad is not None:
                    p += params[n].grad ** 2
            self.model.zero_grad()


        for n, p in fisher.items():
            eps_tensor = torch.tensor(self.eps).to(p.device)
            max_val = torch.maximum(p / len(train_loader), eps_tensor)
            fisher[n] = max_val.detach().clone()

        inv_fisher = {n: 1 / p.detach().clone() for n, p in fisher.items()}
        inv_sum_fisher = torch.cat([p.detach().ravel() for p in fisher.values()]).mean()
        final_fisher = {n: 1 / (p.detach().clone() * inv_sum_fisher) for n, p in fisher.items()}
        self.model.train()
        return final_fisher, inv_fisher


    def after_update(self, **kwargs: Any):
        super().after_update(**kwargs)  # type: ignore

        # self.model.clamp_radii()

    def after_training_exp(self, **kwargs: Any):
        super().after_training_exp(**kwargs)
        fisher, self.inv_fisher = self.compute_fisher()

    def before_training_exp(self, **kwargs: Any):
        super().before_training_exp(**kwargs)  # type: ignore

        if self.training_exp_counter == 1:
            self.make_optimizer()
            # TODO: adaptive choice of scale?
            self.apply_fisher_scaling(
                    self.inv_fisher, self.robust_loss_threshold, first_time=True)
            self.model.switch_mode(Mode.CONTRACTION)
            self.make_optimizer()
        elif self.training_exp_counter > 1:
            self.model.freeze_task()
            self.model.switch_mode(Mode.VANILLA)
            self.apply_fisher_scaling(
                    self.inv_fisher, self.robust_loss_threshold, first_time=False)
            self.model.switch_mode(Mode.CONTRACTION)
            self.make_optimizer()



    def before_training_epoch(self, **kwargs: Any):
        super().before_training_epoch(**kwargs)  # type: ignore

        # if self.mode == Mode.VANILLA and self.vanilla_loss is not None \
        #         and self.vanilla_loss < self.vanilla_loss_threshold:
        #     self.make_optimizer()
        #     self.model.switch_mode(Mode.EXPANSION)

        if self.viz_debug:
            if self.windows.get('accuracy'):
                # Reset plots every epoch
                self.viz_debug.line(X=torch.tensor([0]), Y=torch.tensor([0]),
                                    win=self.windows['accuracy'], name='acc')
                self.viz_debug.line(X=torch.tensor([0]), Y=torch.tensor([0]),
                                    win=self.windows['accuracy'], name='robust_acc')
            else:
                self.windows['accuracy'] = self.viz_debug.line(
                    X=torch.tensor([0]),
                    Y=torch.tensor([0]),
                    win=None,
                    opts={'title': 'Batch accuracy'}
                )

    def robust_output(self):
        if self.regression_task:
            output_lower, _, output_higher = self.mb_output_all['last'].unbind('bounds')
            lower_error = ((output_lower - self.mb_y) ** 2).rename(None)
            higher_error = ((output_higher - self.mb_y) ** 2).rename(None)
            return torch.where(higher_error > lower_error, output_higher.rename(None), output_lower.rename(None))
        else:
            output_lower, _, output_higher = self.mb_output_all['last'].unbind('bounds')
            y_oh = F.one_hot(self.mb_y)  # type: ignore
            return torch.where(y_oh.bool(), output_lower.rename(None), output_higher.rename(None))  # type: ignore
