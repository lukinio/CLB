GPUID=$1
OUTDIR=outputs/split_CIFAR100_cnn_interval_id
REPEAT=5
mkdir -p $OUTDIR

#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 10 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --clipping --eps_per_model --batch_size 100 --lr 0.001 \
#       --reg_coef 10 \
#       --schedule 150 \
#       --kappa_epoch 150 \
#       --eps_epoch 150 \
#       --eps_val 2 1 0.5 0.25 0.12 0.1 0.1 0.1 0.1 0.1 --eps_max 0 \
#       | tee ${OUTDIR}/tn1_id.log


#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 10 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --clipping --eps_per_model --batch_size 100 --lr 0.001 \
#       --reg_coef 10 \
#       --schedule 150 \
#       --kappa_epoch 100 \
#       --eps_epoch 150 \
#       --eps_val 2.5 1.2 0.6 0.3 0.2 0.2 0.1 0.1 0.1 0.1 --eps_max 0 \
#       | tee ${OUTDIR}/tn2_id.log
#
#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 10 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --clipping --eps_per_model --batch_size 100 --lr 0.001 \
#       --reg_coef 10 \
#       --schedule 200 \
#       --kappa_epoch 150 \
#       --eps_epoch 200 \
#       --eps_val 2.5 1.2 0.6 0.3 0.2 0.2 0.1 0.1 0.1 0.1 --eps_max 0 \
#       | tee ${OUTDIR}/tn3_id.log

#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 10 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --clipping --eps_per_model --batch_size 100 --lr 0.001 \
#       --reg_coef 10 \
#       --schedule 200 \
#       --kappa_epoch 150 \
#       --eps_epoch 200 \
#       --eps_val 2.5 1.2 0.6 0.5 0.5 0.5 0.4 0.4 0.3 0.2 --eps_max 0 \
#       | tee ${OUTDIR}/tn3_id.log


#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 10 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --clipping --eps_per_model --batch_size 100 --lr 0.001 \
#       --reg_coef 10 \
#       --schedule 200 \
#       --kappa_epoch 150 \
#       --eps_epoch 200 \
#       --eps_val 2.5 1.2 1 0.8 0.6 0.4 0.4 0.3 0.3 0.2 --eps_max 0 \
#       | tee ${OUTDIR}/tn4_id.log


#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 10 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --clipping --eps_per_model --batch_size 100 --lr 0.001 \
#       --reg_coef 10 --schedule 150 --kappa_epoch 100 --eps_epoch 150 \
#       --eps_val 3 2.8 2.6 2.4 2.2 2 1.8 1.6 1.4 1.2 \
#       --eps_max 0 \
#       | tee ${OUTDIR}/tn100_6_id.log


#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 10 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --clipping --eps_per_model --batch_size 100 --lr 0.001 \
#       --schedule 150 --kappa_epoch 140 --eps_epoch 150 \
#       --eps_val 2.5 2.1 1.7 1.4 1 1 0.8 0.6 0.4 0.2 \
#       --eps_max 0 \
#       | tee ${OUTDIR}/id_20.log


#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 10 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --clipping --eps_per_model --batch_size 100 --lr 0.001 \
#       --schedule 150 --kappa_epoch 140 --eps_epoch 150 \
#       --eps_val 3 2.1 1.8 1.4 1 0.8 0.5 0.4 0.2 0.1 \
#       --eps_max 0 \
#       | tee ${OUTDIR}/id_201.log

#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 10 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --clipping --eps_per_model --batch_size 100 --lr 0.001 \
#       --schedule 150 --kappa_epoch 140 --eps_epoch 150 \
#       --eps_val 3 2.1 1.8 1.4 1.1 0.8 0.6 0.5 0.4 0.3 \
#       --eps_max 0 \
#       | tee ${OUTDIR}/id_202.log


#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 10 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --clipping --eps_per_model --batch_size 100 --lr 0.001 \
#       --schedule 150 --kappa_epoch 140 --eps_epoch 150 \
#       --eps_val 3 2.1 1.8 1.4 1.1 1.1 1.1 1 0.95 0.9 \
#       --eps_max 0 \
#       | tee ${OUTDIR}/id_203.log


#python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 10 --first_split_size 10 \
#       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
#       --agent_name IntervalNet --clipping --eps_per_model --batch_size 100 --lr 0.001 \
#       --schedule 150 --kappa_epoch 120 --eps_epoch 150 \
#       --eps_val 0.1 --kappa_min 0 \
#       --eps_max 0 \
#       | tee ${OUTDIR}/id_204.log

python -u intervalBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID \
       --repeat $REPEAT --optimizer Adam --force_out_dim 10 --first_split_size 10 \
       --other_split_size 10 --model_name interval_cnn --model_type cnn --agent_type interval \
       --agent_name IntervalNet --clipping --eps_per_model --batch_size 100 --lr 0.001 \
       --schedule 150 --kappa_epoch 100 --eps_epoch 150 \
       --eps_val 0.01 --kappa_min 0 \
       --eps_max 0 \
       | tee ${OUTDIR}/id_205.log