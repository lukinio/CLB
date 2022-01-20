#!/bin/sh

#SBATCH --job-name=interval_cl
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task 5

trap "kill 0" INT

# TODO