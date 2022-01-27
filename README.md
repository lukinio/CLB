# IntervalNet

## Setup

Setup conda environment:

```console
conda env create -f environment.yml
```

Populate `.env` file with settings from `.env.example`, e.g.:

```
DATA_DIR=~/datasets
RESULTS_DIR=~/results
WANDB_ENTITY=some_entity
WANDB_PROJECT=intervalnet_cl
```

## Training

Default interval training experiment:

```console
python train.py cfg=default cfg.seed=2001
```

Scripts for recreating the experiments from the paper are in the `scripts` directory. 


## Development

Main interval training logic is in `intervalnet/strategy.py`. Model specifics are in `intervalnet/models/interval.py`.