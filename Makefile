.PHONY: train

notes=

train:
	@python train.py notes="$(notes)"

benchmark-interval:
	@for SEED in 1234 1235 1236 1237 1238; do python train.py 'tags=[Interval,benchmark]' group=benchmark-interval cfg.seed=$$SEED; done

benchmark-sgd:
	@for SEED in 1234 1235 1236 1237 1238; do python train.py group=benchmark-sgd cfg=sgd cfg.seed=$$SEED; done
	
benchmark-adam:
	@for SEED in 1234 1235 1236 1237 1238; do python train.py group=benchmark-adam cfg=adam cfg.seed=$$SEED; done

benchmark-ewc:
	@for SEED in 1234 1235 1236 1237 1238; do python train.py group=benchmark-ewc cfg=ewc cfg.seed=$$SEED; done

debug:
	@python train.py cfg=sgd cfg.scenario=INC_TASK 'notes=inc task debug' cfg.epochs=3
	@python train.py cfg=sgd cfg.scenario=INC_DOMAIN 'notes=inc domain debug' cfg.epochs=3
	@python train.py cfg=sgd cfg.scenario=INC_CLASS 'notes=inc class debug' cfg.epochs=3
	