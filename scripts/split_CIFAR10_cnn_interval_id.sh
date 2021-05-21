GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_interval_id_test
REPEAT=2
mkdir -p $OUTDIR


# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#        --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#        --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#        --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#        --eps_val 1 0.8 0.6 0.4 0.2 --eps_epoch 140 --eps_max 1 0.8 0.6 0.4 0.2 \
#        --kappa_epoch 140 --schedule 150 --kappa_min 0  \
#        | tee ${OUTDIR}/in_pw_test6.log


# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#        --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#        --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#        --agent_name IntervalNet --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#        --eps_val 1 0.8 0.6 0.4 0.2 --eps_epoch 40 --eps_max 1 0.8 0.6 0.4 0.2 \
#        --kappa_epoch 30 --schedule 50 --kappa_min 0  \
#        | tee ${OUTDIR}/in_pw_test7.log

# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#        --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#        --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#        --agent_name IntervalNet --batch_size 64 --lr 0.001 --clipping --eps_per_model \
#        --eps_val 1 0.8 0.6 0.4 0.2 --eps_epoch 40 --eps_max 1 0.8 0.6 0.4 0.2 \
#        --kappa_epoch 40 --schedule 50 --kappa_min 0  \
#        | tee ${OUTDIR}/in_pw_experimental_old_robustloss_code.log

python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
       --agent_name IntervalNet --batch_size 64 --lr 0.001 --clipping --eps_per_model \
       --eps_val 10000 --eps_epoch 40 --eps_max 10000 \
       --kappa_epoch 40 --schedule 50 --kappa_min 1.0  \
       | tee ${OUTDIR}/in_pw_experimental_no_robustloss.log

# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#        --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#        --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#        --agent_name IntervalNet --batch_size 128 --lr 0.001 --clipping --eps_per_model \
#        --eps_val 100 80 60 40 20 --eps_epoch 40 --eps_max 100 80 60 40 20 \
#        --kappa_epoch 40 --schedule 50 --kappa_min 0.5  \
#        | tee ${OUTDIR}/in_pw_experimental_kappa_1.log