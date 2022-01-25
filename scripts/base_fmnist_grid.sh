#!/bin/sh

trap "kill 0" INT

# Fashion MNIST
    
for seed in 2001; do
    for scenario in INC_DOMAIN; do
        common="cfg.seed=${seed} cfg.scenario=${scenario} tags=["${scenario}"] cfg.dataset=FASHION_MNIST"
        
        for lr in 0.001 0.005 0.0005 0.01 0.0001; do
            echo "Running SGD ${common} lr=${lr}"
            python train.py cfg=baseline_sgd group=fmnist_grid_sgd cfg.learning_rate=${lr} ${common} > /dev/null 2>&1 &
        done
        wait

        for lr in 0.001 0.005 0.0005 0.01 0.0001; do
            echo "Running ADAM ${common} lr=${lr}"
            python train.py cfg=baseline_adam group=fmnist_grid_adam cfg.learning_rate=${lr} ${common} > /dev/null 2>&1 &
        done
        wait

        for lambda in 0.1 1 10 100 0.001; do
            echo "Running L2 ${common} lambda=${lambda}"
            python train.py cfg=baseline_l2 group=fmnist_grid_l2 cfg.ewc_lambda=${lambda} ${common} > /dev/null 2>&1 &
        done
        wait

        for lambda in 1 16 256 1024 2048 4096 8192; do
            echo "Running EWC ${common} lambda=${lambda}"
            python train.py cfg=baseline_ewc cfg.ewc_labmda=${lambda} group=fmnist_grid_ewc ${common} > /dev/null 2>&1 &
        done
        wait

        for lambda in 1 16 256 1024 2048 4096 8192; do
            echo "Running Online EWC ${common} lambda=${lambda}"
            python train.py cfg=baseline_ewc group=fmnist_grid_ewco cfg.ewc_mode=online ${common} > /dev/null 2>&1 &
        done
        wait

        for lambda in 1 16 256 1024 2048 4096 8192; do
            echo "Running SI ${common} lambda=${lambda}"
            python train.py cfg=baseline_si cfg.si_lambda=${lambda} group=fmnist_grid_si ${common} > /dev/null 2>&1 &
        done
        wait

        for lambda in 0.1 0.5 1 2 4; do
            echo "Running MAS ${common} lambda=${lambda}"
            python train.py cfg=baseline_mas group=fmnist_grid_mas cfg.ewc_lambda=${lambda} ${common} > /dev/null 2>&1 &
        done

        for temperature in 0.5 1 2; do
            for alpha in 0.25 0.5 1 2 8; do
                echo "Running LWF ${common} temp=${temperature} alpha=${alpha}"
                python train.py cfg=baseline_lwf group=fmnist_grid_lwf ${common} > /dev/null 2>&1 &
            done
            wait
        done

        for lr in 0.5 0.1  0.05 0.01 0.005 0.001 0.0005 0.0001; do
            echo "Running joint ${common} lr=${lr}"
            python train.py cfg=baseline_joint cfg.learning_rate=${lr} group=fmnist_grid_joint ${common} > /dev/null 2>&1 &
        done 
        wait

    done
    
    wait
done

