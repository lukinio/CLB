GPUID=$1
OUTDIR=outputs/split_CIFAR100_cnn_interval_id
REPEAT=1
mkdir -p $OUTDIR


python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --optimizer Adam --force_out_dim 10 --first_split_size 10 --other_split_size 10 --schedule 150 \
       --model_name interval_cnn --model_type cnn --agent_type interval --agent_name IntervalNet \
       --batch_size 100 --lr 0.001 --kappa_epoch 150 --eps_epoch 150 --eps_val 0.1 --eps_max 0.1 \
       | tee ${OUTDIR}/Adam_1_test.log


#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 10 --first_split_size 10 --other_split_size 10 --schedule 100 \
#       --model_name interval_cnn --model_type cnn --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --kappa_epoch 100 --eps_epoch 100 --eps_val 0.1 --eps_max 0.1 \
#       | tee ${OUTDIR}/Adam_01.log
#
#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 10 --first_split_size 10 --other_split_size 10 --schedule 100 \
#       --model_name interval_cnn --model_type cnn --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --kappa_epoch 100 --eps_epoch 100 --eps_val 0.1 --eps_max 0.1 \
#       --clipping | tee ${OUTDIR}/Adam_01_clipping.log