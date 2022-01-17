#!/bin/sh

trap "kill 0" INT

# SGD
for cfg in adam; do
    for seed in 2001 2002 2003 2004 2005; do
        for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
            for lr in 0.1 0.05 0.01 0.005 0.001; do
                python train.py cfg=baseline_${cfg} group=${cfg} cfg.seed=${seed} cfg.learning_rate=${lr} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                echo "Running ${cfg} on ${scenario} with seed = ${seed}, lr = ${lr}"
            done
        wait
        done
    done
done