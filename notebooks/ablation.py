import gc
import os
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import imageio
import ipywidgets as widgets
import matplotlib
import numpy as np
import plotly
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.cuda
import torch.distributions
import torch.nn as nn
import torch.optim
from agents.interval import IntervalNet
from colour import Color
from IPython.display import HTML, display
from ipywidgets import Box, GridBox, HBox, IntSlider, Output, VBox
from pandas import Interval
from plotly.graph_objs._figure import Figure
from rich import print
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.notebook import tqdm

# FONT_FAMILY = 'Nunito'
FONT_FAMILY = 'Roboto Condensed'

bgcolor = '#F0F4F7'
bgcolor2 = '#D2D6D9'

red = '#D81B60'
red_bg = '#E6CFD7'
blue = '#1E88E5'
blue_bg = '#D3E0EB'
yellow = '#FFC107'
yellow_bg = '#FFF9E6'
green = '#008C0B'
green_bg = '#CFE6D0'


def cuda_mem(device: Optional[str] = None):
    if device == 'cpu':
        return
    if device is None:
        device = 'cuda:0'

    alloc = torch.cuda.memory_allocated(device) / 2**20  # type: ignore
    max_alloc = torch.cuda.max_memory_allocated(device) / 2**20  # type: ignore
    reserved = torch.cuda.memory_reserved(device) / 2**20  # type: ignore

    print(f'[bold white]'
          f'Alloc: [yellow]{alloc:7.1f} MB[white]  '
          f'MaxAlloc: [yellow]{max_alloc:7.1f} MB[white]  '
          f'Reserved: [yellow]{reserved:7.1f} MB[white]  '
          f'[red]\\[{device}][white] '
          )


def _hex_to_rgba(hex: str, a: float):
    assert 0 <= a <= 1
    r, g, b = [int(round(c * 255)) for c in Color(hex).rgb]
    return f'rgba({r}, {g}, {b}, {a})'


class Points2D(Dataset):
    def __init__(self, n_points: int = 12, n_classes: int = 2, low: float = 0.05, high: float = 0.95, radius: float = 0.16, seed: int = 1, task_id: int = 0):
        self.n_points = n_points
        self.n_classes = n_classes
        self.low = low
        self.high = high
        self.radius = radius
        self.seed = seed
        self.task_id = task_id
        self.rand: np.random.RandomState = np.random.RandomState(seed=seed)
        self.epsilon: float
        self.coords: torch.Tensor
        self.labels: torch.Tensor
        # np.random.seed(seed)

        self._generate_points()

    def _new_point(self):
        return self.rand.uniform(self.low, self.high, size=2)

    def _generate_points(self):
        points = [self._new_point()]

        while (len(points) < self.n_points):
            p = self._new_point()
            if min(np.abs(p - p_).sum() for p_ in points) > 2 * self.radius:
                points.append(p)

        self.epsilon = self.radius / 2

        coords = np.array(points, dtype=np.float32)
        labels = np.array(self.rand.randint(0, self.n_classes, size=self.n_points))

        self.coords = torch.from_numpy(coords)
        self.labels = torch.from_numpy(labels)

    def __getitem__(self, index):
        return self.coords[index], self.labels[index], torch.Tensor(self.task_id)

    def __len__(self):
        return len(self.coords)


def plot(points: Points2D, model=None, title: str = '', low: float = 0., high: float = 1., moves: List[float] = None, hires: bool = True):
    labels = np.array(points.labels)
    n_classes = len(np.unique(labels))
    assert n_classes <= 4

    labels = labels.tolist()

    fig = go.Figure()  # type: ignore

    symbols = ['diamond', 'circle', 'hexagon', 'square']
    colors = [blue, yellow, red, green]
    colors_bg = [
        [0 / (n_classes - 1), _hex_to_rgba(blue_bg, 1)],
        [1 / (n_classes - 1), _hex_to_rgba(yellow_bg, 1)],
        [2 / (n_classes - 1), _hex_to_rgba(red_bg, 1)],
        [3 / (n_classes - 1), _hex_to_rgba(green_bg, 1)],
    ][:n_classes]

    fig.add_trace(go.Scatter(  # type: ignore
        x=points.coords[:, 0].numpy(),
        y=points.coords[:, 1].numpy(),
        mode='markers',
        marker=dict(
            size=10,
            symbol=[symbols[c] for c in labels],
            cmin=-0.5,
            cmax=1.5,
            color=[colors[c] for c in labels],
            line=dict(
                width=1,
                color='rgba(0, 0, 0, 0.5)',
            )
        ),
        showlegend=False,
    ))

    fig.update_layout(
        plot_bgcolor=bgcolor,
        xaxis=dict(
            range=[low, high],
            nticks=2,
            showticklabels=False,
            showline=True,
            linewidth=1,
            linecolor=bgcolor2,
            mirror=True,
            showgrid=False,
            zeroline=False,
            title=''
        ),
        yaxis=dict(
            range=[low, high],
            nticks=2,
            showticklabels=False,
            showline=True,
            linewidth=1,
            linecolor=bgcolor2,
            mirror=True,
            showgrid=False,
            zeroline=False,
            title=''
        ),
        title=dict(
            text=f'<b>{title}</b>',
            font=dict(
                size=12,
            ),
        ),
        font=dict(
            family=FONT_FAMILY,
        ),
        width=400,
        height=425,
        margin=dict(
            l=25,
            r=25,
            t=50,
            b=25,
            pad=5,
            autoexpand=False
        ),
        autosize=False,
    )

    if model is not None:
        model.eval()
        xrange = np.linspace(low, high, 600 if hires else 200)
        yrange = np.linspace(low, high, 600 if hires else 200)
        xx, yy = np.meshgrid(xrange, yrange)

        inputs = torch.Tensor([xx.ravel(), yy.ravel()]).T
        if isinstance(model, IntervalNet):
            zz = model.predict(inputs.cuda())['All'].argmax(dim=1).view(xx.shape).detach().cpu().numpy()
        else:
            zz = model(inputs).argmax(dim=1).view(xx.shape).detach().numpy()

        fig.add_trace(go.Contour(  # type: ignore
            x=xrange,
            y=yrange,
            z=zz,
            colorscale=colors_bg,
            zmin=0,
            zmax=n_classes - 1,
            autocontour=False,
            contours=dict(
                type='levels',
                start=0.5,
                end=n_classes - 1.5,
                size=1,
                coloring='fill',
                showlines=True,
                showlabels=False,
            ),
            showlegend=False,
            showscale=False,
            opacity=1.0,
            line=dict(
                color='rgba(0.5, 0.5, 0.5, 1.0)',
                width=1.5,
                smoothing=0.0,
                # dash='solid' if moves else 'dot',
                dash='solid',
            )
        ))

        if isinstance(model, IntervalNet) and moves is not None and n_classes == 2:
            # scale = px.colors.diverging.Tealrose
            scale = px.colors.diverging.Tropic
            c_low, c_mid, c_high = scale[0], scale[len(scale) // 2], scale[-1]

            lines = {}

            for move in moves:
                model.save_params()
                model.move_weights(move)
                zz = model.predict(inputs.cuda())['All'].argmax(dim=1).view(xx.shape).detach().cpu().numpy()
                model.restore_weights()

                if move < 0:
                    color = plotly.colors.find_intermediate_color(c_low, c_mid, intermed=1 + move, colortype='rgb')
                else:
                    color = plotly.colors.find_intermediate_color(c_mid, c_high, intermed=move, colortype='rgb')

                # Class fill
                fig.add_trace(go.Contour(  # type: ignore
                    x=xrange,
                    y=yrange,
                    z=zz,
                    colorscale=colors_bg,
                    zmin=0,
                    zmax=n_classes - 1,
                    autocontour=False,
                    contours=dict(
                        type='levels',
                        start=0.5,
                        end=n_classes - 1.5,
                        size=1,
                        coloring='fill',
                        showlines=False,
                        showlabels=False,
                    ),
                    showscale=False,
                    opacity=0.25,
                ))

                # Epsilon boundaries
                eps = {
                    '1.00': ' + ε',
                    '0.50': ' + 0.5ε',
                    '-0.50': ' - 0.5ε',
                    '-1.00': ' - ε',
                }[f'{move:.2f}']
                name = f'<b>W<sub>k</sub>{eps}</b>'
                fig.add_trace(go.Contour(  # type: ignore
                    x=xrange,
                    y=yrange,
                    z=zz,
                    colorscale=colors_bg,
                    zmin=0,
                    zmax=n_classes - 1,
                    autocontour=False,
                    contours=dict(
                        start=0.5,
                        end=n_classes - 1.5,
                        size=1,
                        coloring='none',
                    ),
                    name=name,
                    legendgroup=name,
                    showscale=False,
                    opacity=1.0,
                    line=dict(
                        color=color,
                        width=1.5,
                        dash='dot',
                    )
                ))

                lines[move] = (name, color)

            # Strong border for mid prediction
            zz = model.predict(inputs.cuda())['All'].argmax(dim=1).view(xx.shape).detach().cpu().numpy()
            fig.add_trace(go.Contour(  # type: ignore
                x=xrange,
                y=yrange,
                z=zz,
                colorscale=colors_bg,
                zmin=0,
                zmax=n_classes - 1,
                autocontour=False,
                contours=dict(
                    start=0.5,
                    end=n_classes - 1.5,
                    size=1,
                    coloring='none',
                ),
                name=f'<b>W<sub>k</sub></b>',
                showscale=False,
                opacity=1.0,
                line=dict(
                    color='rgba(0.5, 0.5, 0.5, 1.0)',
                    width=1.5,
                    dash='solid',
                )
            ))

            # Custom legend
            fig.update_layout(
                showlegend=False,
            )
            # legend_x = 0.8
            # legend_y = 0.03
            # legend_w = 0.2
            # legend_h = 0.25
            # fig.add_shape(  # type: ignore
            #     type='rect',
            #     xref='x', yref='y',
            #     x0=legend_x, y0=legend_y,
            #     x1=legend_x + legend_w, y1=legend_y + legend_h,
            #     fillcolor=bgcolor,
            #     line_color=bgcolor2,
            #     line_width=1,
            # )

            lines[0] = ('<b>W<sub>k</sub></b>', 'rgba(0.5, 0.5, 0.5, 1.0)')
            moves.append(0)

            for i, move in enumerate(sorted(moves)):
                fig.add_annotation(  # type: ignore
                    text=lines[move][0],
                    font_size=11,
                    x=0.86,
                    y=0.27 - i * 0.05,
                    xanchor='left',
                    yanchor='middle',
                    showarrow=False,
                    textangle=0,
                )

                fig.add_trace(go.Scatter(  # type: ignore
                    x=[0.82, 0.86],
                    y=[0.27 - i * 0.05, 0.27 - i * 0.05],
                    mode='lines',
                    line=dict(
                        color=lines[move][1],
                        width=1.5,
                        dash='dot' if move != 0 else 'solid',
                    )
                ))

    return fig


@ dataclass
class Plot:
    start: int = 50
    end: int = 500
    step: int = 50

    plots: Dict[int, Optional[Figure]] = field(init=False)
    ui: VBox = field(init=False)

    def __post_init__(self):
        self.plots = {epoch: None for epoch in range(self.start, self.end + 1, self.step)}

        self.out = Output()
        self.slider = IntSlider(min=self.start, max=self.end, step=self.step, continuous_update=False,
                                layout={'margin': 'auto'})
        self.slider.observe(self._on_epoch_change, names='value')

        self.ui = VBox([  # type: ignore
            self.out,
            self.slider
        ], layout={'border': '1px solid #D2D6D9', 'width': '400px'})

    def _on_epoch_change(self, change):
        self.update()

    def update(self):
        self.out.clear_output(wait=True)
        with self.out:
            if self.plots[self.slider.value] is not None:
                self.plots[self.slider.value].show(config=dict(displayModeBar=False))

    def select(self, epoch: int):
        self.slider.value = epoch

    def add(self, epoch: int, fig: Figure):
        self.plots[epoch] = fig
        if self.slider.value != epoch:
            self.select(epoch)
        else:
            self.update()


# https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/20
def parameterized_truncated_normal(uniform, mu, sigma, a, b):
    normal = torch.distributions.normal.Normal(0, 1)

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    alpha_normal_cdf = normal.cdf(alpha)
    p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * uniform

    p = p.numpy()
    one = np.array(1, dtype=p.dtype)
    epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
    v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
    x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
    x = torch.clamp(x, a, b)

    return x


def truncated_normal(uniform):
    return parameterized_truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2)


def sample_truncated_normal(shape=()):
    return truncated_normal(torch.from_numpy(np.random.uniform(0, 1, shape)))


def train_plot_mlp(mlp, mlp_plot, points, max_epochs=500, low=0, high=1):
    torch.manual_seed(1)
    # 1, 3, 42, 1024

    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         nn.init.trunc_normal_(m.weight, std=2 / np.sqrt(m.in_features))

    # mlp.linear.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    # opt = torch.optim.SGD(mlp.parameters(), lr=0.01)

    t = tqdm(range(1, max_epochs + 1))
    for epoch in t:
        mlp.train()
        out = mlp(points.coords)
        loss = criterion(out, points.labels)

        accuracy = (out.argmax(dim=1) == points.labels).sum().item() / len(points.coords)
        t.set_description(desc=f'{epoch} --> {np.round(accuracy * 100, 2)}')

        if epoch % mlp_plot.step == 0:
            fig = plot(points, mlp, high=high, low=low,
                       title=f'Vanilla MLP --> epoch: {epoch}, accuracy: {np.round(accuracy * 100, 2)}%')
            mlp_plot.add(epoch, fig)

            fig.write_image(f'figures/vanilla_{epoch}.png', width=600, height=600)

        opt.zero_grad()
        loss.backward()
        opt.step()


def train_interval(agent, dataloader, kappa_goal=1.0, kappa_goal_epoch=1, eps_goal=0, eps_goal_epoch=50):
    iters_per_epoch = len(dataloader)

    Schedule = namedtuple('Schedule', ['start', 'goal', 'goal_epoch'])

    kappa = Schedule(
        start=1.0,
        goal=kappa_goal,
        goal_epoch=kappa_goal_epoch,
    )
    eps = Schedule(
        start=0,
        goal=eps_goal,
        goal_epoch=eps_goal_epoch,
    )

    agent.kappa_scheduler.set_end(kappa.goal)
    agent.kappa_scheduler.calc_coefficient(kappa.goal - kappa.start, kappa.goal_epoch, iters_per_epoch)
    agent.kappa_scheduler.current = kappa.start

    agent.eps_scheduler.set_end(0)  # no limit
    agent.eps_scheduler.calc_coefficient(eps.goal - eps.start, eps.goal_epoch, iters_per_epoch)
    agent.eps_scheduler.current = eps.start

    agent.learn_batch(dataloader)


def plot_interval(agent, interval_plot, points, epoch=0, move: float = 0.0, low=0, high=1):
    if move != 0.:
        agent.save_params()
        agent.move_weights(move)

    out = agent(points.coords.cuda())['All'].cpu()
    accuracy = (out.argmax(dim=1) == points.labels).sum().item() / len(points.coords)

    fig = plot(
        points, agent, title=f'Interval MLP ({move}) --> epoch: {epoch}, accuracy: {np.round(accuracy * 100, 2)}%',
        low=low, high=high
    )
    interval_plot.add(epoch, fig)

    fig.write_image(f'figures/interval_{epoch}_{move:.2f}.png', width=600, height=600)

    if move != 0:
        agent.restore_weights()


def plot_interval_condensed(agent, interval_plot, points, epoch=500, low=0, high=1):
    fig = plot(points, agent, title=f'Decision boundary of an interval multi-layer perceptron',
               moves=[-1, -0.5, 0.5, 1], low=low, high=high)
    interval_plot.add(epoch, fig)


def plot_interval_animation(agent, points, low=0, high=1):
    agent.save_params()

    os.makedirs('figures/interval_animation', exist_ok=True)
    with imageio.get_writer(
        'figures/interval_animation/interval.gif',
        mode='I',
        duration=0.1,
    ) as writer:

        moves = np.hstack([np.linspace(-1, 0, 50)[:-1], np.linspace(0, 1, 50)])

        for i, move in enumerate(tqdm(moves)):
            agent.move_weights(move)

            out = agent(points.coords.cuda())['All'].cpu()
            accuracy = (out.argmax(dim=1) == points.labels).sum().item() / len(points.coords)

            title = (
                f'Interval MLP, base weights {"+" if move >= 0 else ""}{move:.2f}'
                f' epsilon --> accuracy: {np.round(accuracy * 100, 2)}%'
            )
            fig = plot(points, agent, title=title, low=low, high=high)

            filename = f'figures/interval_animation/{i}.png'
            fig.write_image(filename, width=600, height=600)

            image = imageio.imread(filename)
            writer.append_data(image)

            if move == 0 or move == -1 or move == 1:
                for _ in range(10):
                    writer.append_data(image)

            agent.restore_weights()
