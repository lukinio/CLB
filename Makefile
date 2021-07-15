.PHONY: train

notes=

train:
	@NOTES="$(notes)" python src/train.py
