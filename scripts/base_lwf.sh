#!/bin/sh

trap "kill 0" INT

# Learning without Forgetting
for cfg in lwf; do
    for scenario in INC_DOMAIN; do
        for seed in 2001; do
            for temperature in 0.5 2 1; do
                for alpha in 0.25 0.5 1 2 8 16; do
                    python train.py cfg=baseline_${cfg} group=${cfg}_final cfg.optimizer=ADAM cfg.seed=${seed} cfg.lwf_alpha=${alpha} cfg.lwf_temperature=${temperature} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
                    echo "Running ${cfg} on ${scenario} with seed = ${seed}, alpha = ${alpha}, temp. = ${temperature}"
                done
                wait
            done
        done
    done
    
    # for scenario in INC_DOMAIN INC_CLASS INC_TASK; do
    #     for temperature in 0.5; do
    #         for alpha in 0.5; do
    #             for seed in 2001 2002 2003 2004 2005; do
    #                 python train.py cfg=baseline_${cfg} group=${cfg}_final cfg.optimizer=SGD cfg.seed=${seed} cfg.lwf_alpha=${alpha} cfg.lwf_temperature=${temperature} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
    #                 echo "Running ${cfg} on ${scenario} with seed = ${seed}, alpha = ${alpha}, temp. = ${temperature}"
    #             done
    #             wait
    #         done
    #     done

    #     for temperature in 1; do
    #         for alpha in 2; do
    #             for seed in 2001 2002 2003 2004 2005; do
    #                 python train.py cfg=baseline_${cfg} group=${cfg}_final cfg.optimizer=SGD cfg.seed=${seed} cfg.lwf_alpha=${alpha} cfg.lwf_temperature=${temperature} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
    #                 echo "Running ${cfg} on ${scenario} with seed = ${seed}, alpha = ${alpha}, temp. = ${temperature}"
    #             done
    #             wait
    #         done
    #     done

    #     for temperature in 2; do
    #         for alpha in 8; do
    #             for seed in 2001 2002 2003 2004 2005; do
    #                 python train.py cfg=baseline_${cfg} group=${cfg}_final cfg.optimizer=SGD cfg.seed=${seed} cfg.lwf_alpha=${alpha} cfg.lwf_temperature=${temperature} cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
    #                 echo "Running ${cfg} on ${scenario} with seed = ${seed}, alpha = ${alpha}, temp. = ${temperature}"
    #             done
    #             wait
    #         done
    #     done

    # done
done