#!/bin/sh

trap "kill 0" INT

# EWC
for cfg in ewc; do
    for seed in 2001 2002 2003 2004 2005; do
        for lambda in 1e1 1e2 1e3 1e4 1e5 1e6 1e7 1e8 1e9 1e10; do
            for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
                python train.py cfg=baseline_${cfg} group=${cfg} cfg.seed=${seed} cfg.optimizer=ADAM cfg.reg_lambda=${lambda} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                echo "Running ${cfg} on ${scenario} with seed = ${seed}, lambda = ${lambda}, Adam"
            done
            wait
        done
    done
done