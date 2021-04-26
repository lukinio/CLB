GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_interval_id
REPEAT=1
mkdir -p $OUTDIR


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 50 \
#       --model_name interval_cnn --model_type cnn --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --kappa_epoch 25 --eps_epoch 50 --eps_val 0.8 --eps_max 0.8 \
#       | tee ${OUTDIR}/Adam_1_test.log

# good 71
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 50 \
#       --model_name interval_cnn --model_type cnn --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --kappa_epoch 25 --eps_epoch 50 --eps_val 0.5 --eps_max 0.5 \
#       | tee ${OUTDIR}/Adam_last.log

python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 50 \
       --model_name interval_cnn --model_type cnn --agent_type interval --agent_name IntervalNet \
       --batch_size 100 --lr 0.001 --kappa_epoch 25 --eps_epoch 50 --eps_val 1 --eps_max 1 \
       | tee ${OUTDIR}/Adam_last1.log


