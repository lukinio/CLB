# IntervalNet

## Setup

Setup conda environment:

```console
conda env create -f environment.yml
```

Install additional requirements:

```
pip install git+https://github.com/pytorch/hydra-torch/#subdirectory=hydra-configs-torch git+https://github.com/pytorch/hydra-torch/#subdirectory=hydra-configs-torchvision
```

Populate `.env` file with settings from `.env.example`, e.g.:

```
DATA_DIR=~/datasets
RESULTS_DIR=~/results
WANDB_ENTITY=gmum
WANDB_PROJECT=intervalnet_test
```

Make sure that `pytorch-yard` is using the appropriate version (defined in `train.py`).
If not, then correct package version with something like:

```console
pip install --force-reinstall pytorch-yard==2021.10.11
```

## Training

Default interval training experiment:

```console
python train.py cfg=interval cfg.seed=2001
```

## Development

Main interval training logic is in `intervalnet/strategy.py`. Model specifics are in `intervalnet/models/interval.py`.

## Various

For typing with Avalanche add `py.typed` to the Avalanche package root directory.
