.PHONY: train

notes=

train:
	@python train.py notes="$(notes)"

benchmark-interval:
	@python train.py -m group=benchmark-interval 'cfg.seed=range(0,5)'

benchmark-sgd:
	@python train.py -m group=benchmark-sgd cfg=sgd 'cfg.seed=range(0,5)'

benchmark-adam:
	@python train.py -m group=benchmark-adam cfg=adam 'cfg.seed=range(0,5)'