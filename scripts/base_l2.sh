#!/bin/sh

trap "kill 0" INT

# EWC
for cfg in ewc; do
    for optimizer in SGD; do
        for lambda in 0.1; do
            for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
                for seed in 2001 2002 2003 2004 2005; do
                    python train.py cfg=baseline_l2 group=l2_final cfg.seed=${seed} cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                    echo "Running L2 on ${scenario} with seed = ${seed}, lambda = ${lambda}, ${optimizer}"
                done
            done
            wait
        done
    done

    for optimizer in ADAM; do
        for lambda in 10; do
            for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
                for seed in 2001 2002 2003 2004 2005; do
                    python train.py cfg=baseline_l2 group=l2_final cfg.seed=${seed} cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                    echo "Running L2 on ${scenario} with seed = ${seed}, lambda = ${lambda}, ${optimizer}"
                done
            done
            wait
        done
    done

done