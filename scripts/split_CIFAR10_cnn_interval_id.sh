GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_interval_id_test
REPEAT=2
mkdir -p $OUTDIR

# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#        --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#        --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
#        --agent_name IntervalNet --batch_size 64 --lr 0.001 --clipping --eps_per_model \
#        --eps_val 10000 --eps_epoch 40 --eps_max 10000 \
#        --kappa_epoch 40 --schedule 50 --kappa_min 1.0  \
#        | tee ${OUTDIR}/in_pw_experimental_no_robustloss.log

python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
       --agent_name IntervalNet --batch_size 128 --lr 0.001 --clipping --eps_per_model \
      --eps_val 100000 --eps_epoch 100 --eps_max 100000 \
       --kappa_epoch 100 --schedule 100 --kappa_min 0.0  \
       | tee ${OUTDIR}/in_pw_experimental.log