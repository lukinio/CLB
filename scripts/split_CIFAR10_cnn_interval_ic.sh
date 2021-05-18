GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_interval_ic
REPEAT=10
mkdir -p $OUTDIR

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
#       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
#       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 128 \
#       --eps_val 2 2 0.5 0.5 0.1 --eps_epoch 50 --eps_max 0 \
#       --kappa_epoch 30 --schedule 50 \
#        | tee ${OUTDIR}/t1.log


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
#       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
#       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 128 \
#       --eps_val 2.5 2 2 2 2 --eps_epoch 60 --eps_max 0 \
#       --kappa_epoch 20 --schedule 60 \
#        | tee ${OUTDIR}/t1.log

# 21-22
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
#       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
#       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 100 \
#       --eps_val 2 --eps_epoch 60 --eps_max 0 \
#       --kappa_epoch 20 --schedule 60 \
#        | tee ${OUTDIR}/t2.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
#       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
#       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 50 \
#       --eps_val 0.1 0.05 0.05 0.05 0.05 --eps_epoch 50 --eps_max 0 \
#       --kappa_epoch 10 --schedule 50 \
#        | tee ${OUTDIR}/t3.log


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
#       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
#       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 100 \
#       --eps_val 1 0.9 0.8 0.7 0.6 --eps_epoch 50 --eps_max 0 \
#       --kappa_epoch 40 --schedule 50 \
#        | tee ${OUTDIR}/t4.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
#       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
#       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 100 \
#       --eps_val 1 0.95 0.95 0.9 0.8 --eps_epoch 60 --eps_max 0 \
#       --kappa_epoch 40 --schedule 60 \
#        | tee ${OUTDIR}/t5.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
#       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
#       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 100 \
#       --eps_val 0.5 --eps_epoch 50 --eps_max 0 \
#       --kappa_epoch 20 --schedule 50 \
#        | tee ${OUTDIR}/t6.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
#       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
#       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 100 \
#       --eps_val 0.5 --eps_epoch 50 --eps_max 0 \
#       --kappa_epoch 20 --schedule 50 \
#        | tee ${OUTDIR}/t7.log
#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
#       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
#       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 100 \
#       --eps_val 8 5 5 5 5 --eps_epoch 80 --eps_max 0 \
#       --kappa_epoch 50 --schedule 80 --weight_decay 5e-4 \
#        | tee ${OUTDIR}/t8.log


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
#       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
#       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 100 \
#       --eps_val 0.05 --eps_epoch 50 --eps_max 0 \
#       --kappa_epoch 10 --schedule 50 \
#        | tee ${OUTDIR}/t10.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
#       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
#       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 100 \
#       --eps_val 0.5 0.4 0.3 0.2 0.1 --eps_epoch 40 --eps_max 0.5 0.4 0.3 0.2 0.1 \
#       --kappa_epoch 30 --schedule 50 --weight_decay 5e-4 \
#        | tee ${OUTDIR}/t9.log

python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 100 \
       --eps_val 0.01 --eps_epoch 50 --eps_max 0 \
       --kappa_epoch 40 --schedule 50 \
        | tee ${OUTDIR}/t11.log




