#!/bin/sh

trap "kill 0" INT

# Synaptic Intelligence
for cfg in si; do
    for seed in 2001 2002 2003 2004 2005; do
        for lambda in 0.25 1024 0.5 1 2 128 4 8 256 16 32 64 512; do
            for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
                python train.py cfg=baseline_${cfg} group=${cfg} cfg.seed=${seed} cfg.si_lambda=${lambda} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                echo "Running ${cfg} on ${scenario} with seed = ${seed}, lambda = ${lambda}"
            done
            wait
        done
    done
done