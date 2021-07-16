import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Type

import hydra
import setproctitle
from dotenv import load_dotenv
from omegaconf import OmegaConf

from .configs import RootConfig, Settings, register_configs
from .utils.logging import info, info_bold
from .utils.rundir import setup_rundir

load_dotenv()


class Experiment(ABC):
    def __init__(self, config_path: str, settings_cls: Type[Settings], settings_group: Optional[str] = None) -> None:
        """
        Run an experiment from a provided entry point with minimal boilerplate code.

        Incorporates run directory setup and Hydra support.
        """

        self.settings_cls = settings_cls
        self.settings_group = settings_group

        self.cfg: RootConfig
        self._initialize: Callable[[], None]

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

        register_configs(self.settings_cls, self.settings_group)

        self._initialize = self._pytorch

        hydra_decorator = hydra.main(config_path=config_path, config_name='root')
        hydra_decorator(self.main)()

    @abstractmethod
    def main(self, cfg: RootConfig):
        """
        Main experiment function. Called after initial setup with `cfg` populated by Hydra.

        Parameters
        ----------
        cfg : Config
            Top-level Hydra config for the experiment.
        """
        self.cfg = cfg
        self._initialize()

    def _pytorch(self):
        RUN_NAME = os.getenv('RUN_NAME')

        info_bold(f'\\[init] Run name --> {RUN_NAME}')
        info(f'\\[init] Loaded config:\n{OmegaConf.to_yaml(self.cfg, resolve=True)}')

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
        # tags = get_tags(cast(DictConfig, cfg))
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
