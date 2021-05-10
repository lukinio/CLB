GPUID=$1
OUTDIR=outputs/split_CIFAR100_cnn_interval_ic
REPEAT=5
mkdir -p $OUTDIR

#
#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --incremental_class --optimizer Adam --force_out_dim 100 --no_class_remap \
#       --first_split_size 10 --other_split_size 10 --model_name interval_cnn --model_type cnn \
#       --agent_type interval  --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --clipping --eps_per_model --schedule 150 --kappa_epoch 100 --eps_epoch 150 \
#       --eps_val 2 1.9 1.8 1.7 1.6 1.5 1.4 1.3 1.2 1.1 \
#       --eps_max 0 \
#       | tee ${OUTDIR}/ic.log

python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT \
       --incremental_class --optimizer Adam --force_out_dim 100 --no_class_remap \
       --first_split_size 10 --other_split_size 10 --model_name interval_cnn --model_type cnn \
       --agent_type interval  --agent_name IntervalNet --lr 0.001 --batch_size 100 \
       --clipping --eps_per_model --schedule 150 --kappa_epoch 140 --eps_epoch 150 \
       --eps_val 2.5 2.45 2.4 2.35 2.3 2.25 2.2 2.15 2.1 2.05 \
       --eps_max 0 \
       | tee ${OUTDIR}/ic1.log
