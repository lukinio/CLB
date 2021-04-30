import os
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, Optional

import ipywidgets as widgets
import matplotlib
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
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

FONT_FAMILY = 'Nunito'

bgcolor = '#F0F4F7'
bgcolor2 = '#D2D6D9'

red = '#D81B60'
red_bg = '#D1A3B4'
blue = '#1E88E5'
blue_bg = '#B7D3EB'
yellow = '#FFC107'


def _hex_to_rgba(hex: str, a: float):
    assert 0 <= a <= 1
    r, g, b = [int(round(c * 255)) for c in Color(hex).rgb]
    return f'rgba({r}, {g}, {b}, {a})'


class Points2D(Dataset):
    def __init__(self, n_points: int = 12, low: float = 0.05, high: float = 0.95, radius: float = 0.16, seed: int = 1, task_id: int = 0):
        self.n_points = n_points
        self.low = low
        self.high = high
        self.radius = radius
        self.seed = seed
        self.task_id = task_id
        self.rand: np.random.Generator = np.random.default_rng(seed=seed)
        self.epsilon: float
        self.coords: torch.Tensor
        self.labels: torch.Tensor

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
        labels = np.array(np.round(self.rand.uniform(0, 1, size=self.n_points)), dtype=np.int64)

        self.coords = torch.from_numpy(coords)
        self.labels = torch.from_numpy(labels)

    def __getitem__(self, index):
        return self.coords[index], self.labels[index], torch.Tensor(self.task_id)

    def __len__(self):
        return len(self.coords)


def plot(points: Points2D, model=None, title: str = '', low: float = 0., high: float = 1.):
    labels = np.array(points.labels).tolist()

    fig = go.Figure()  # type: ignore

    colors = [red, blue]
    symbols = ['diamond', 'circle']

    fig.add_trace(go.Scatter(  # type: ignore
        x=points.coords[:, 0],
        y=points.coords[:, 1],
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
        )
    ))

    fig.update_layout(
        plot_bgcolor=bgcolor,
        xaxis=dict(
            range=[0, 1],
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
            range=[0, 1],
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
        xrange = np.linspace(low, high, 400)
        yrange = np.linspace(low, high, 400)
        xx, yy = np.meshgrid(xrange, yrange)

        inputs = torch.Tensor([xx.ravel(), yy.ravel()]).T
        if isinstance(model, IntervalNet):
            zz = model(inputs.cuda())['All'].argmax(dim=1).view(xx.shape).detach().cpu().numpy()
        else:
            zz = model(inputs).argmax(dim=1).view(xx.shape).detach().numpy()

        fig.add_trace(go.Contour(  # type: ignore
            x=xrange,
            y=yrange,
            z=zz,
            colorscale=[
                [0, _hex_to_rgba(red_bg, 1.0)],
                [1, _hex_to_rgba(blue_bg, 1.0)]
            ],
            showscale=False,
            opacity=1.0,
            line=dict(
                color='rgba(0, 0, 0, 0.05)',
                width=1.0,
                dash='dot',
            )
        ))

    return fig


@dataclass
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
