from dataclasses import dataclass

import pytorch_yard


# Experiment settings validation schema & default values
@dataclass
class Settings(pytorch_yard.Settings):
    # ----------------------------------------------------------------------------------------------
    # General experiment settings
    # ----------------------------------------------------------------------------------------------
    batch_size: int = 32
