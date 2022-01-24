#!/bin/sh

trap "kill 0" INT

# Fashion MNIST
    
for seed in 2001; do
    for scenario in INC_DOMAIN; do
        
        # # SGD
        python train.py cfg=baseline_sgd cfg.dataset=FASHION_MNIST group=fmnist_sgd cfg.seed=${seed} cfg.optimizer=SGD cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
        echo "Running SGD on ${scenario} with seed = ${seed}"

        # # Adam
        python train.py cfg=baseline_adam cfg.dataset=FASHION_MNIST group=fmnist_adam cfg.seed=${seed} cfg.optimizer=ADAM cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
        echo "Running Adam on ${scenario} with seed = ${seed}"
    
        # Offline SGD
        python train.py cfg=baseline_joint cfg.dataset=FASHION_MNIST group=fmnist_joint_sgd cfg.seed=${seed} cfg.optimizer=SGD cfg.learning_rate=0.5 cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
        echo "Running offline SGD on ${scenario} with seed = ${seed}"

        # Offline Adam
        python train.py cfg=baseline_joint cfg.dataset=FASHION_MNIST group=fmnist_joint_adam cfg.seed=${seed} cfg.optimizer=ADAM cfg.learning_rate=0.001 cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
        echo "Running offline Adam on ${scenario} with seed = ${seed}"

        # EWC SGD
        python train.py cfg=baseline_ewc cfg.dataset=FASHION_MNIST group=fmnist_ewco cfg.seed=${seed} cfg.optimizer=SGD cfg.learning_rate=0.001 cfg.ewc_lambda=2048 cfg.ewc_mode=online cfg.scenario=${scenario} tags=["${scenario}"] > /dev/null 2>&1 &
        echo "Running online EWC SGD on ${scenario} with seed = ${seed}"

        wait

    done
done

