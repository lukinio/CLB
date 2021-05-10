GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_interval_ic
REPEAT=2
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

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
#       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
#       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 100 \
#       --eps_val 2 1.5 1.5 1.5 0.5 --eps_epoch 50 --eps_max 2 1.5 1.5 1.5 0.5 \
#       --kappa_epoch 50 --schedule 100 \
#        | tee ${OUTDIR}/ic1.log



python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
       --incremental_class --optimizer Adam --force_out_dim 10 --no_class_remap \
       --first_split_size 2 --other_split_size 2 --model_name interval_cnn \
       --model_type cnn --clipping --eps_per_model --lr 0.001 --batch_size 100 \
       --eps_val 1 1 1 0.9 0.8 --eps_epoch 50 --eps_max 1 1 1 0.9 0.8 \
       --kappa_epoch 30 --schedule 60 \
        | tee ${OUTDIR}/ic2.log




