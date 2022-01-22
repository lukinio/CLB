#!/bin/sh

#SBATCH --job-name=interval_cl
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task 5

trap "kill 0" INT

# TODO
offline=False
scenario=INC_TASK

idx=0
for elr in 1.; do
	export CUDA_VISIBLE_DEVICES=${idx}
	python train.py cfg=default_cifar100 cfg.scenario=${scenario} cfg.offline=${offline} \
		cfg.strategy=Interval cfg.interval.expansion_learning_rate=${elr} cfg.interval.robust_lambda=1. \
		tags=["20220120_interval_cifar100_it"] &
	idx=$((idx+1))
done
