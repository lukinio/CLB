#!/bin/sh

trap "kill 0" INT

# Joint training
for cfg in joint; do
    for seed in 2001 2002 2003 2004 2005; do
        for optimizer in SGD; do
            for lr in 0.05 0.005 0.0005 0.0001; do
                for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
                        python train.py cfg=baseline_${cfg} group=${cfg}_${optimizer} cfg.optimizer=${optimizer} cfg.seed=${seed} cfg.learning_rate=${lr} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                        echo "Running ${cfg} on ${scenario} with seed = ${seed}, optimizer = ${optimizer}, lr = ${lr}"
                done
                wait
            done
            for lr in 1 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001; do
                for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
                        python train.py cfg=baseline_${cfg} group=${cfg}_${optimizer} cfg.optimizer=${optimizer} cfg.momentum=0.9 cfg.seed=${seed} cfg.learning_rate=${lr} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                        echo "Running ${cfg} on ${scenario} with seed = ${seed}, optimizer = ${optimizer}, lr = ${lr}"
                done
                wait
            done
        done
        for optimizer in ADAM; do
            for lr in 0.05 0.005 0.0005 0.0001; do
                for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
                        python train.py cfg=baseline_${cfg} group=${cfg}_${optimizer} cfg.optimizer=${optimizer} cfg.seed=${seed} cfg.learning_rate=${lr} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                        echo "Running ${cfg} on ${scenario} with seed = ${seed}, optimizer = ${optimizer}, lr = ${lr}"
                done
                wait
            done
        done
    done
done