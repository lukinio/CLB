#!/bin/sh

trap "kill 0" INT

for dataset in MNIST FASHION_MNIST; do    
    for seed in 2001 2002 2003 2004 2005; do
        for scenario in INC_DOMAIN INC_CLASS INC_TASK; do
            common="cfg.dataset=${dataset} cfg.seed=${seed} cfg.scenario=${scenario} tags=["${scenario}"]"
            
            echo "Running SGD ${common}"
            python train.py cfg=baseline_sgd group=${dataset}_sgd ${common} > /dev/null 2>&1 &

            echo "Running ADAM ${common}"
            python train.py cfg=baseline_adam group=${dataset}_adam ${common} > /dev/null 2>&1 &

            echo "Running L2 ${common}"
            python train.py cfg=baseline_l2 group=${dataset}_l2 ${common} > /dev/null 2>&1 &

            echo "Running EWC ${common}"
            python train.py cfg=baseline_ewc group=${dataset}_ewc ${common} > /dev/null 2>&1 &
            
            echo "Running Online EWC ${common}"
            python train.py cfg=baseline_ewc group=${dataset}_ewco cfg.ewc_mode=online ${common} > /dev/null 2>&1 &

            echo "Running SI ${common}"
            python train.py cfg=baseline_si group=${dataset}_si ${common} > /dev/null 2>&1 &

            echo "Running MAS ${common}"
            python train.py cfg=baseline_mas group=${dataset}_mas ${common} > /dev/null 2>&1 &

            echo "Running LWF ${common}"
            python train.py cfg=baseline_lwf group=${dataset}_lwf ${common} > /dev/null 2>&1 &
        
            echo "Running joint ${common}"
            if [ "${dataset}" = "MNIST" ]; then
                python train.py cfg=baseline_joint cfg.strategy=Naive cfg.offline=True cfg.learning_rate=0.5 group=${dataset}_joint ${common} > /dev/null 2>&1 &
            else
                python train.py cfg=baseline_joint cfg.strategy=Naive cfg.offline=True group=${dataset}_joint ${common} > /dev/null 2>&1 &
            fi
        
            wait

        done
    done
done