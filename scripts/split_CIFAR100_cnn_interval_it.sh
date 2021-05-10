GPUID=$1
OUTDIR=outputs/split_CIFAR100_cnn_interval_it
REPEAT=5
mkdir -p $OUTDIR

#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --schedule 100  --batch_size 100 --lr 0.001 --kappa_epoch 100 \
#       --eps_epoch 100 --eps_val 10 5 2.5 1.25 0.625 0.3 0.15 0.1 0.1 0.1 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/Adam_test.log


#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --schedule 150  --batch_size 128 --lr 0.001 --kappa_epoch 100 \
#       --eps_epoch 150 --eps_val 20 10 5 2.5 1.25 0.6 0.3 0.2 0.1 0.1 --eps_max 0 \
#       --clipping --reg_coef 10 --milestones 80 120 | tee ${OUTDIR}/Adam_test1.log

#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --schedule 150  --batch_size 128 --lr 0.001 --kappa_epoch 100 \
#       --eps_epoch 150 --eps_val 2 1 0.5 0.25 0.12 0.1 0.1 0.1 0.1 0.1 --eps_max 0 --eps_per_model \
#       --clipping --reg_coef 10 | tee ${OUTDIR}/tn1.log

#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --schedule 150  --batch_size 128 --lr 0.001 --kappa_epoch 100 \
#       --eps_epoch 150 --eps_val 2 1 0.5 0.25 0.12 0.1 0.1 0.1 0.1 0.1 --eps_max 0 --eps_per_model \
#       --clipping --reg_coef 10 | tee ${OUTDIR}/tn11.log



#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --schedule 150  --batch_size 128 --lr 0.001 --kappa_epoch 100 \
#       --eps_epoch 150 --eps_val 2 1.9 1.8 1.7 1.6 1.5 1.4 1.3 1.2 1.1 --eps_max 0 --eps_per_model \
#       --clipping --reg_coef 100 | tee ${OUTDIR}/tn100_6.log

#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --schedule 160  --batch_size 100 --lr 0.001 --kappa_epoch 100 \
#       --eps_epoch 150 --eps_val 2 1.2 0.8 0.6 0.6 0.5 0.5 0.4 0.4 0.3 \
#       --eps_max 2 1.2 0.8 0.6 0.6 0.5 0.5 0.4 0.4 0.3 --eps_per_model \
#       --clipping --reg_coef 100 | tee ${OUTDIR}/tn100_7.log
#
#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --schedule 150  --batch_size 100 --lr 0.001 --kappa_epoch 100 \
#       --eps_epoch 150 --eps_val 2 1.2 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 \
#       --eps_max 0 --eps_per_model \
#       --clipping --reg_coef 100 | tee ${OUTDIR}/tn100_8.log

#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --schedule 150  --batch_size 100 --lr 0.001 --clipping \
#       --eps_epoch 150 --eps_val 2.5 1.6 1.2 1.1 1 0.9 0.8 0.7 0.6 0.5 \
#       --eps_max 0 --kappa_epoch 100 --eps_per_model \
#       | tee ${OUTDIR}/it_60.log

# Current best
#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --schedule 150 --batch_size 100 --lr 0.001 --clipping \
#       --eps_epoch 150 --eps_val 2.5 1.9 1.5 1.2 1.1 1 0.9 0.9 0.8 0.8 \
#       --eps_max 0 --kappa_epoch 140 --eps_per_model \
#       | tee ${OUTDIR}/it_60_1.log

#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --schedule 150 --batch_size 100 --lr 0.001 --clipping \
#       --eps_epoch 150 --eps_val 2.5 1.9 1.5 1.2 1.1 1 0.9 0.8 0.7 0.6 \
#       --eps_max 0 --kappa_epoch 140 --eps_per_model \
#       | tee ${OUTDIR}/it_60_2.log

python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" \
       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 10 \
       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
       --agent_name IntervalNet --schedule 150 --batch_size 100 --lr 0.001 --clipping \
       --eps_epoch 150 --eps_val 2.5 1.9 1.5 1.2 1.1 1 0.8 0.6 0.4 0.3 \
       --eps_max 0 --kappa_epoch 140 --eps_per_model \
       | tee ${OUTDIR}/it_60_3.log


#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid "${GPUID}" \
#       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 0 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --schedule 150  --batch_size 128 --lr 0.001 --kappa_epoch 100 \
#       --eps_epoch 150 --eps_val 2 1 0.5 0.5 0.3 0.3 0.2 0.2 0.1 0.1 --eps_max 0 --eps_per_model \
#       --clipping | tee ${OUTDIR}/tn100_5.log
