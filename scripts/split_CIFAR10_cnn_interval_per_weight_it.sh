GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_interval_per_weight_it
REPEAT=1
mkdir -p $OUTDIR


# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#        --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#        --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#        --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#        --kappa_epoch 10 --eps_epoch 50 --eps_val 20 6.6 3.3 1.08 0.35 --eps_max 0 \
#        --clipping | tee ${OUTDIR}/test25.log

# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#        --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#        --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#        --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#        --kappa_epoch 20 --eps_epoch 50 --eps_val 1 0 0 0 0 --eps_max 0 \
#        --clipping | tee ${OUTDIR}/test31.log


# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#        --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#        --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#        --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#        --kappa_epoch 20 --eps_epoch 50 --eps_val 3 1 0.5 0.3 0.1 --eps_max 0 \
#        --clipping | tee ${OUTDIR}/test32.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 30 --eps_epoch 50 --eps_val 10 3.3 1.1 0.36 0.12 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test33.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 30 --eps_epoch 50 --eps_val 10 3.3 1.1 0.36 0.12 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test331.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 30 --eps_epoch 50 --eps_val 1 0 1 0 1 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test34.log
#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 30 --eps_epoch 50 --eps_val 1 0 0 1 1 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test35.log
#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 30 --eps_epoch 50 --eps_val 1 0 0 0 1 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test36.log
#

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 50 \
#       --kappa_epoch 30 --eps_epoch 50 --eps_val 10 3.3 1.1 0.36 0.12 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test37.log

#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 12 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 4 --eps_epoch 12 --eps_val 0.5 0 0 0 0.5 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test371.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 12 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 4 --eps_epoch 12 --eps_val 0 0 0 0 0 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test3712.log


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 24 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 4 --eps_epoch 12 --eps_val 0.1 0.1 0.1 0.1 0.1 --eps_max 0 \
#       --clipping --weight_decay 3e-4 | tee ${OUTDIR}/test3713.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 20 --eps_epoch 20 --eps_val 500 333.3 111.1 36 10 --eps_max 0 \
#       --schedule 20 --clipping --weight_decay 3e-4 | tee ${OUTDIR}/test37145.log

# GOOOD
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 20 --eps_epoch 20 --eps_val 100 33.3 11.1 3.6 0 --eps_max 0 \
#       --schedule 20 --clipping | tee ${OUTDIR}/test37146.log

# GOOOD
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 20 --eps_epoch 20 --eps_val 100 33.3 11.1 3.6 1.2 --eps_max 0 \
#       --schedule 20 --clipping | tee ${OUTDIR}/in_pw_zeros.log

python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
       --other_split_size 2 --model_name interval_cnn --model_type cnn \
       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
       --kappa_epoch 50 40 30 20 20 --eps_epoch 1 1 1 1 1 --eps_val 10 3.3 1.1 0.36 0.12 --eps_max 0 \
       --schedule 80 80 60 40 40 --clipping | tee ${OUTDIR}/test37148_rand.log


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 120 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 40 --eps_epoch 120 --eps_val 30 10 1 0.1 0.1 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test38.log
#
#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 240 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 80 --eps_epoch 240 --eps_val 100 0 0 0 0 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test39.log




#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 12 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 4 --eps_epoch 12 --eps_val 0.1 0.1 0.1 0.1 0.1 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test42.log
#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 12 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 4 --eps_epoch 12 --eps_val 0.9 0.4 0.2 0.1 0.1 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test43.log
#
#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 12 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 8 --eps_epoch 12 --eps_val 0.1 0.1 0.1 0.1 0.1 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test44.log
#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 12 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 8 --eps_epoch 12 --eps_val 0.9 0.4 0.2 0.1 0.1 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test45.log
#
#
#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 20 --eps_epoch 50 --eps_val 0.1 0.1 0.1 0.1 0.1 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test40.log
#
#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 20 --eps_epoch 50 --eps_val 0.9 0.4 0.2 0.1 0.1 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test41.log


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 20 --eps_epoch 50 --eps_val 0.9 0.4 0.2 0.1 0.1 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test42.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 240 \
#       --kappa_epoch 80 --eps_epoch 240 --eps_val 100 0 0 0 0 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test37.log



# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#        --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#        --other_split_size 2 --schedule 120 --model_name interval_cnn --model_type cnn \
#        --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#        --kappa_epoch 40 --eps_epoch 120 --eps_val 100 10 3 1 0.4 --eps_max 0 \
#        --clipping | tee ${OUTDIR}/test32.log
