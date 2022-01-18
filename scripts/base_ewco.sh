#!/bin/sh

trap "kill 0" INT

# EWC online
for cfg in ewc; do
    for seed in 2001 2002 2003 2004 2005; do
        for optimizer in SGD; do
            for lambda in 2048 1024 4096 512 8196 128 256 16384 64 32 16 1 2 4 8; do
                for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
                    python train.py cfg=baseline_${cfg} group=${cfg}_online cfg.seed=${seed} cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} cfg.ewc_mode=online cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                    echo "Running ${cfg}_online on ${scenario} with seed = ${seed}, lambda = ${lambda}, ${optimizer}"
                done
                wait
            done
        done
    done
done