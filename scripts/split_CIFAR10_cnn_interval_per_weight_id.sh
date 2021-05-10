GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_interval_id_per_weight
REPEAT=10
mkdir -p $OUTDIR


# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#        --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#        --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#        --agent_type interval --agent_name IntervalNet --batch_size 100 --lr 0.001 \
#        --kappa_epoch 25 --eps_epoch 50 --eps_val 10 3.3 1.1 0.36 0.12 --eps_max 0 \
#        --clipping | tee ${OUTDIR}/test1.log


# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#        --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#        --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#        --agent_type interval --agent_name IntervalNet --batch_size 100 --lr 0.001 \
#        --kappa_epoch 20 --eps_epoch 50 --eps_val 10 3.3 1.1 0.36 0.12 --eps_max 0 \
#        --clipping | tee ${OUTDIR}/test2.log


# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#        --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#        --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#        --agent_type interval --agent_name IntervalNet --batch_size 100 --lr 0.001 \
#        --kappa_epoch 30 --eps_epoch 50 --eps_val 10 3.3 1.1 0.36 0.12 --eps_max 0 \
#        --clipping | tee ${OUTDIR}/test3.log

# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#        --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#        --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#        --agent_type interval --agent_name IntervalNet --batch_size 100 --lr 0.001 \
#        --kappa_epoch 40 --eps_epoch 50 --eps_val 10 3.3 1.1 0.36 0.12 --eps_max 0 \
#        --clipping | tee ${OUTDIR}/test4.log

# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#        --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#        --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#        --agent_type interval --agent_name IntervalNet --batch_size 100 --lr 0.001 \
#        --kappa_epoch 50 --eps_epoch 50 --eps_val 10 3.3 1.1 0.36 0.12 --eps_max 0 \
#        --clipping | tee ${OUTDIR}/test5.log


# python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#        --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#        --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#        --agent_type interval --agent_name IntervalNet --batch_size 100 --lr 0.001 \
#        --kappa_epoch 50 --eps_epoch 50 --eps_val 30 10 3 0.5 0.1 --eps_max 0 \
#        --clipping | tee ${OUTDIR}/test6.log



#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#       --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --batch_size 100 --lr 0.001 \
#       --kappa_epoch 30 --eps_epoch 50 --eps_val 10 3.3 1.1 0.36 0.12 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test8.log


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#       --other_split_size 2 --schedule 2 --model_name interval_cnn --model_type cnn \
#       --agent_type interval --agent_name IntervalNet --batch_size 100 --lr 0.001 \
#       --kappa_epoch 40 --eps_epoch 1 --eps_val 100 10 3 1 0.4 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test88.log

 python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
        --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
        --other_split_size 2 --schedule 50 --model_name interval_cnn --model_type cnn \
        --agent_type interval --agent_name IntervalNet --batch_size 100 --lr 0.001 \
        --kappa_epoch 50 --eps_epoch 50 --eps_val 10 3.3 1.1 0.36 0.12 --eps_max 0 \
        --clipping | tee ${OUTDIR}/test_id.log
