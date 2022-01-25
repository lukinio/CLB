#!/bin/sh

trap "kill 0" INT

# Fashion MNIST
    
for seed in 2001 2002 2003 2004 2005; do
    for scenario in INC_DOMAIN INC_CLASS INC_TASK; do
        common="cfg.seed=${seed} cfg.scenario=${scenario} tags=["${scenario}"] cfg.dataset=FASHION_MNIST"
        
        echo "Running SGD ${common}"
        python train.py cfg=baseline_sgd group=fmnist_sgd ${common} > /dev/null 2>&1 &

        echo "Running ADAM ${common}"
        python train.py cfg=baseline_adam group=fmnist_adam ${common} > /dev/null 2>&1 &

        echo "Running L2 ${common}"
        python train.py cfg=baseline_l2 group=fmnist_l2 ${common} > /dev/null 2>&1 &

        echo "Running EWC ${common}"
        python train.py cfg=baseline_ewc group=fmnist_ewc ${common} > /dev/null 2>&1 &
        
        echo "Running Online EWC ${common}"
        python train.py cfg=baseline_ewc group=fmnist_ewco cfg.ewc_mode=online ${common} > /dev/null 2>&1 &

        echo "Running SI ${common}"
        python train.py cfg=baseline_si group=fmnist_si ${common} > /dev/null 2>&1 &

        echo "Running MAS ${common}"
        python train.py cfg=baseline_mas group=fmnist_mas ${common} > /dev/null 2>&1 &

        echo "Running LWF ${common}"
        python train.py cfg=baseline_lwf group=fmnist_lwf ${common} > /dev/null 2>&1 &
       
        echo "Running joint ${common}"
        python train.py cfg=baseline_joint group=fmnist_joint ${common} > /dev/null 2>&1 &
       
        # wait

    done
    
    wait
done

