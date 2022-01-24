#!/bin/sh

trap "kill 0" INT

# MAS
for cfg in mas; do

    # for optimizer in SGD; do
    #     for seed in 2001; do
    #         for lambda in 1 2 4 8 16 32 64 128 256; do
    #             for scenario in INC_DOMAIN; do
    #                 python train.py cfg=baseline_mas group=mas_grid cfg.seed=${seed} cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
    #                 echo "Running MAS on ${scenario} with seed = ${seed}, lambda = ${lambda}, ${optimizer}"
    #             done
    #         done
    #         wait
    #     done
    # done

    for optimizer in SGD; do
        for seed in 2001; do
            for lambda in 512 1024 2048 4096 8196 16384; do
                for scenario in INC_DOMAIN; do
                    python train.py cfg=baseline_mas group=mas_grid cfg.seed=${seed} cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                    echo "Running MAS on ${scenario} with seed = ${seed}, lambda = ${lambda}, ${optimizer}"
                done
            done
            wait
        done
    done

done