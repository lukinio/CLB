.PHONY: train

notes=

train:
	@WANDB=on NOTES="$(notes)" ./scripts/split_CIFAR10_cnn_interval_id.sh 0

ewc:
	python -u iBatchLearn.py --dataset CIFAR10 --train_aug --gpuid 0 --repeat 1 --optimizer Adam \
			--force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 50 --batch_size 128 \
			--model_name cnn --model_type cnn --agent_type customization  --agent_name EWC \
			--lr 0.001 --reg_coef 10       | tee outputs/split_CIFAR10_cnn_base/EWC_id_save.log

fisher:
	python -u intervalBatchLearn.py --gpuid 0 --repeat 1 --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
		   --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet --batch_size 128 --lr 0.001 --clipping --eps_val 1140800 --eps_mode sum --gradient_clipping 1 --reg_coef 10