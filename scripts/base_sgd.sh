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

# cifar LR search
for lr in 0.1 0.05 0.01 0.005 0.001; do
  python train.py cfg=ic_cifar100 cfg.offline=True cfg.learning_rate=${lr}
done