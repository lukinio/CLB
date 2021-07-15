import os
from datetime import datetime
from functools import partial, update_wrapper
from pathlib import Path
from typing import Callable, cast

import coolname
import hydra
import setproctitle
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from .configs import Config, get_tags, register_configs
from .utils.logging import info
from .utils.rundir import setup_rundir

load_dotenv()


MainFunc = Callable[[Config], None]


def _pytorch(cfg: Config, main_func: MainFunc):
    RUN_NAME = os.getenv('RUN_NAME')

    info(f'\\[init] Run name --> {RUN_NAME}')
    # log.info(f'\\[init] Loaded config:\n{OmegaConf.to_yaml(cfg, resolve=True)}')
    main_func(cfg)

    # pl.seed_everything(cfg.experiment.seed)

    #   run: Run = wandb_logger.experiment  # type: ignore

    # # Prepare data using datamodules
    # # https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#using-a-datamodule
    # datamodule: LightningDataModule = instantiate(
    #     cfg.experiment.datamodule,
    #     batch_size=cfg.experiment.batch_size,
    #     seed=cfg.experiment.seed,
    #     shuffle=cfg.experiment.shuffle,
    #     num_workers=cfg.experiment.num_workers
    # )

    # # Create main system (system = models + training regime)
    # system = ImageClassifier(cfg)
    # log.info(f'[bold yellow]\\[init] System architecture:')
    # log.info(system)

    # Setup logging & checkpointing
    tags = get_tags(cast(DictConfig, cfg))
    # run.tags = tags
    # run.notes = str(cfg.notes)
    # wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    # log.info(f'[bold yell

    setproctitle.setproctitle(f'{RUN_NAME} ({os.getenv("WANDB_PROJECT")})')  # type: ignore

    # resume_path = get_resume_checkpoint(cfg, wandb_logger)
    # if resume_path is not None:
    #     log.info(f'[bold yellow]\\[checkpoint] [bold white]{resume_path}')

    # callbacks: list[Any] = []

    # checkpointer = CustomCheckpointer(
    #     period=1,  # checkpointing interval in epochs, but still will save only on validation epoch
    #     dirpath='checkpoints',
    #     filename='{epoch}',
    # )
    # if cfg.experiment.save_checkpoints:
    #     callbacks.append(checkpointer)

    # log.info(f'[bold white]Overriding cfg.pl settings with derived values:')
    # log.info(f' >>> resume_from_checkpoint = {resume_path}')
    # log.info(f' >>> num_sanity_val_steps = {-1 if cfg.experiment.validate_before_training else 0}')
    # log.info(f'')

    # trainer: pl.Trainer = instantiate(
    #     cfg.pl,
    #     logger=wandb_logger,
    #     callbacks=callbacks,
    #     checkpoint_callback=True if cfg.experiment.save_checkpoints else False,
    #     resume_from_checkpoint=resume_path,
    #     num_sanity_val_steps=-1 if cfg.experiment.validate_before_training else 0,
    # )

    # trainer.fit(system, datamodule=datamodule)  # type: ignore
    # # Alternative way to call:
    # # trainer.fit(system, train_dataloader=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())

    # if trainer.interrupted:  # type: ignore
    #     log.info(f'[bold red]>>> Training interrupted.')
    #     run.finish(exit_code=255)


def start(main_func: MainFunc, lightning: bool = False):
    """
    Run an experiment from a provided entry point with minimal boilerplate code.

    `Start` wraps the experiment, incorporating run directory setup and Hydra support.

    Parameters
    ----------
    main_func : MainFunc = Callable[[Config], None]
        Main entry point of the experiment. Should accept a Hydra config as the only argument.
    lightning : bool, optional
        Should PyTorch Lightning support be enabled? Defaults to False (vanilla PyTorch).
    """

    assert os.getenv('DATA_DIR') is not None, "Missing DATA_DIR environment variable."
    assert os.getenv('RESULTS_DIR') is not None, "Missing RESULTS_DIR environment variable."
    assert os.getenv('WANDB_PROJECT') is not None, "Missing WANDB_PROJECT environment variable."

    os.environ['DATA_DIR'] = str(Path(os.environ['DATA_DIR']).expanduser())
    os.environ['RESULTS_DIR'] = str(Path(os.environ['RESULTS_DIR']).expanduser())

    setup_rundir()

    # wandb_logger = WandbLogger(
    #     project=os.getenv('WANDB_PROJECT'),
    #     entity=os.getenv('WANDB_ENTITY'),
    #     name=os.getenv('RUN_NAME'),
    #     save_dir=os.getenv('RUN_DIR'),
    # )

    # # Init logger from source dir (code base) before switching to run dir (results)
    # wandb_logger.experiment  # type: ignore

    register_configs()

    main_variant: Callable[[Config, MainFunc], None] = _pytorch

    main = cast(Callable[[Config], None], partial(main_variant, main_func=main_func))
    update_wrapper(main, main_variant)

    hydra_decorator = hydra.main(config_path='configs', config_name='default')
    hydra_decorator(main)()
