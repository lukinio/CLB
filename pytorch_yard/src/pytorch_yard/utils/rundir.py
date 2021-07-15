import os
from datetime import datetime

import coolname
from dotenv import load_dotenv

load_dotenv()


def setup_rundir():
    """
    Create a separate working directory under `$RESULTS_DIR/$WANDB_PROJECT` with a randomly generated run name.
    """
    date = datetime.now().strftime("%Y%m%d-%H%M")
    name = coolname.generate_slug(2)  # type: ignore
    os.environ['RUN_NAME'] = f'{date}-{name}'

    results_root = f'{os.getenv("RESULTS_DIR")}/{os.getenv("WANDB_PROJECT")}'
    if os.getenv('RUN_MODE', '').lower() == 'debug':
        run_dir = f'{results_root}/_debug/{os.getenv("RUN_NAME")}'
        os.environ['WANDB_MODE'] = 'disabled'
    else:
        run_dir = f'{results_root}/{os.getenv("RUN_NAME")}'

    os.makedirs(run_dir, exist_ok=True)
    os.environ['RUN_DIR'] = run_dir
