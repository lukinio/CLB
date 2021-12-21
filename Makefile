.PHONY: train

notes=

train:
	@python train.py notes="$(notes)"

benchmark-interval:
	@for SEED in 0 1 2 3 4 5; do python train.py group=benchmark-interval cfg.seed=$$SEED; done

benchmark-sgd:
	@for SEED in 0 1 2 3 4 5; do python train.py group=benchmark-sgd cfg=sgd cfg.seed=$$SEED; done
	
benchmark-adam:
	@for SEED in 0 1 2 3 4 5; do python train.py group=benchmark-adam cfg=adam cfg.seed=$$SEED; done