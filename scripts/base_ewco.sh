#!/bin/sh

trap "kill 0" INT

# EWC online
for cfg in ewc; do
    for seed in 2001 2002 2003 2004 2005; do
        for lambda in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8196 16384; do
            for optimizer in SGD ADAM; do
                for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
                    python train.py cfg=baseline_${cfg} group=${cfg}_online cfg.seed=${seed} cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} cfg.ewc_mode=online cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                    echo "Running ${cfg}_online on ${scenario} with seed = ${seed}, lambda = ${lambda}, ${optimizer}"
                done
            done
            wait
        done
    done
done