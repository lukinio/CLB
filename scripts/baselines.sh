#!/bin/sh

trap "kill 0" INT

for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
    for cfg in sgd adam ewc; do
        for seed in 2001 2002 2003 2004 2005; do
            python train.py cfg=baseline_${cfg} group=${scenario}_${cfg} cfg.seed=${seed} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
            echo "Running ${scenario}_${cfg} with seed = $seed"
        done
        wait
    done
done