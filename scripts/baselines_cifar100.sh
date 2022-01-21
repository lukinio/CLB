#!/bin/sh

#SBATCH --job-name=interval_cl
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task 10

trap "kill 0" INT

# SGD
for seed in 2001 2002 2003 2004 2005; do
  for offline in True False; do
    for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
      python train.py cfg=default_cifar100 cfg.seed=${seed} cfg.scenario=${scenario} cfg.offline=${offline} \
        tags=["${scenario}","stdruns"] &
    done
    wait
  done
done

## ADAM
for offline in True False; do
  for seed in 2001 2002 2003 2004 2005; do
    for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
      CUDA_VISIBLE_DEVICES=0 python train.py cfg=default_cifar100 cfg.optimizer=ADAM cfg.learning_rate=0.01 \
        cfg.seed=${seed} cfg.scenario=${scenario} cfg.offline=${offline} \
        tags=["${scenario}","stdruns"] &
    done
    wait
  done
done

# EWC
for seed in 2001 2002 2003 2004 2005; do
  #  for lambda in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8196 16384; do
  for lambda in 2048; do
    #    for optimizer in SGD ADAM; do
    for optimizer in SGD; do
      for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
        python train.py cfg=default_cifar100 cfg.strategy=EWC \
          cfg.seed=${seed} cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} \
          cfg.scenario=${scenario} tags=["${scenario}","stdruns"] &
      done
      wait
    done
  done
done

# EWCOnline
for seed in 2001 2002 2003 2004 2005; do
  #  for lambda in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8196 16384; do
  for lambda in 2048; do
    #    for optimizer in SGD ADAM; do
    for optimizer in SGD; do
      for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
        python train.py cfg=default_cifar100 cfg.strategy=EWC cfg.ewc_mode=online cfg.seed=${seed} \
          cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} \
          cfg.scenario=${scenario} tags=["${scenario}","stdruns"] &
      done
      wait
    done
  done
done

# LWF
for seed in 2001 2002 2003 2004 2005; do
  for temperature in 1; do
    for alpha in 2; do
      for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
        python train.py cfg=default_cifar100 cfg.strategy=LWF cfg.seed=${seed} cfg.lwf_alpha=${alpha} \
          cfg.lwf_temperature=${temperature} cfg.scenario=${scenario} tags=["${scenario}","stdruns"] &
      done
      wait
    done
  done
done

## Synaptic Intelligence
#for seed in 2001; do
#  for lambda in 128; do
#    for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
#      python train.py cfg=default_cifar100 cfg.strategy=SI \
#        cfg.seed=${seed} cfg.si_lambda=${lambda} cfg.scenario=${scenario} tags=["${scenario}","stdruns"] >/dev/null 2>&1 &
#      echo "Running ${cfg} on ${scenario} with seed = ${seed}, lambda = ${lambda}"
#    done
#    wait
#  done
#done
