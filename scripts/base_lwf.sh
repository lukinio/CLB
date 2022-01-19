#!/bin/sh

trap "kill 0" INT

# Learning without Forgetting
for cfg in lwf; do
    # for seed in 2001 2002 2003 2004 2005; do
    for seed in 2001; do
        # for temperature in 2 1 0.5; do
        for temperature in 2 1 0.5; do
            for scenario in INC_DOMAIN; do
                for alpha in 1 0.5 2 0.25 4 8 16; do
                # for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
                    python train.py cfg=baseline_${cfg} group=${cfg}_grid cfg.optimizer=SGD cfg.seed=${seed} cfg.lwf_alpha=${alpha} cfg.lwf_temperature=${temperature} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                    echo "Running ${cfg} on ${scenario} with seed = ${seed}, alpha = ${alpha}, temp. = ${temperature}"
                done
                wait
            done
        done
    done
done