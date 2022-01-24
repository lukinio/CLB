#!/bin/sh

trap "kill 0" INT

# Synaptic Intelligence
for cfg in si; do
    # for seed in 2001; do
    #     for lambda in 0.25 1024 0.5 1 2 128 4 8 256 16 32 64 512; do
    #         for scenario in INC_DOMAIN; do
    #             python train.py cfg=baseline_${cfg} group=${cfg}_grid cfg.optimizer=SGD cfg.seed=${seed} cfg.si_lambda=${lambda} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
    #             echo "Running ${cfg} on ${scenario} with seed = ${seed}, lambda = ${lambda}"
    #         done
    #     done
    #     wait
    # done

    for seed in 2001 2002 2003 2004 2005; do
        for lambda in 2048; do
            for scenario in INC_DOMAIN INC_CLASS INC_TASK; do
                python train.py cfg=baseline_${cfg} group=${cfg}_final cfg.optimizer=SGD cfg.seed=${seed} cfg.si_lambda=${lambda} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                echo "Running ${cfg} on ${scenario} with seed = ${seed}, lambda = ${lambda}"
            done
        done
    done
    wait

done