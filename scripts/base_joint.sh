#!/bin/sh

trap "kill 0" INT

# Joint training
for cfg in joint; do
    for seed in 2001 2002 2003 2004 2005; do
        for lr in 0.001; do
            for optimizer in SGD ADAM; do
                for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
                        python train.py cfg=baseline_${cfg} group=${cfg}_${optimizer} cfg.optimizer=${optimizer} cfg.seed=${seed} cfg.learning_rate=${lr} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                        echo "Running ${cfg} on ${scenario} with seed = ${seed}, optimizer = ${optimizer}, lr = ${lr}"
                    done
                done
            wait
        done
    done
done