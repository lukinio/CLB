import pytorch_yard
from intervalnet import Settings


class Experiment(pytorch_yard.Experiment):
    def main(self, cfg: pytorch_yard.RootConfig):
        super().main(cfg)


if __name__ == '__main__':
    Experiment('intervalnet', Settings)
