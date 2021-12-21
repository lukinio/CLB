.PHONY: train

notes=

train:
	@python train.py notes="$(notes)"

benchmark-sgd:
	@python train.py -m group=benchmark-sgd 'cfg.seed=range(0,5)'

benchmark-adam:
	@python train.py -m group=benchmark-adam 'cfg.seed=range(0,5)'