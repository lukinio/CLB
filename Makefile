.PHONY: train

notes=

train:
	@WANDB=on NOTES="$(notes)" ./scripts/split_CIFAR10_cnn_interval_id.sh 0