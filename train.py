from omegaconf import OmegaConf

from pytorch_yard import Config, start


def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == '__main__':
    start(main)
