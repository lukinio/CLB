#!/bin/sh

trap "kill 0" INT

# MAS
for cfg in mas; do

    for optimizer in SGD; do
        for scenario in INC_DOMAIN INC_TASK INC_CLASS; do
            for lambda in 1; do
                for seed in 2001 2002 2003 2004 2005; do
                    python train.py cfg=baseline_mas group=mas_final cfg.seed=${seed} cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                    echo "Running MAS on ${scenario} with seed = ${seed}, lambda = ${lambda}, ${optimizer}"
                done
                wait
            done
        done
    done

done