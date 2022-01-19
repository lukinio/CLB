#!/bin/sh

trap "kill 0" INT

# SGD
for cfg in sgd; do
    for seed in 2001 2002 2003 2004 2005; do
        for lr in 0.1 0.05 0.01 0.005 0.001; do
            for momentum in 0 0.9; do
                for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
                        python train.py cfg=baseline_${cfg} group=${cfg} cfg.seed=${seed} cfg.momentum=${momentum} cfg.learning_rate=${lr} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                        echo "Running ${cfg} on ${scenario} with seed = ${seed}, momentum = ${momentum}, lr = ${lr}"
                    done
                done
            wait
        done
    done
done

# Standard CIFAR-100 LR search
for lr in 0.1 0.01 1.0 0.001 0.05 0.5 0.005; do
  python train.py cfg=ic_cifar100 cfg.offline=True cfg.learning_rate=${lr} cfg.batchnorm=True tags=["lr_search"]
done

# Interval CIFAR-100 LR search
for lr in 0.1 0.01 1.0 0.001 0.05 0.5 0.005; do
  python train.py cfg=ic_cifar100_interval cfg.offline=True cfg.learning_rate=${lr} cfg.batchnorm=True  tags=["interval","lr_search"]
done

# "borrowed" mobilnet C100 params:
python train.py cfg=ic_cifar100 cfg.offline=True cfg.learning_rate=0.1 cfg.momentum=0.9 cfg.weight_decay=0.0001 cfg.batchnorm=True tags=["weight_decay","schedule","momentum"]
for lr in 0.1 0.01 1.0 0.001 0.05 0.5 0.005 5.0; do
  python train.py cfg=ic_cifar100 cfg.offline=True cfg.learning_rate=${lr} cfg.batchnorm=True tags=["schedule"]
done
