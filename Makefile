.PHONY: train

notes=

train:
	@python train.py notes="$(notes)"
