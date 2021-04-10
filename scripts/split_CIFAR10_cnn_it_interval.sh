GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_interval_it_test
REPEAT=1
mkdir -p $OUTDIR


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 60 --model_name interval_cnn \
#       --model_type cnn --agent_type interval --agent_name IntervalNet --lr 0.001 \
#       --batch_size 100 --kappa_epoch 40 --eps_epoch 60 --eps_val 0.1 --eps_max 0.1 \
#       | tee ${OUTDIR}/Adam_it.log


python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
       --other_split_size 2 --schedule 50 --model_name interval_cnn \
       --model_type cnn --agent_type interval --agent_name IntervalNet --lr 0.001 \
       --batch_size 100 --kappa_epoch 30 --eps_epoch 50 --eps_val 0.2 --eps_max 0.2 \
       | tee ${OUTDIR}/Adam_last.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#       --other_split_size 2 --schedule 60 --model_name interval_cnn \
#       --model_type cnn --agent_type interval --agent_name IntervalNet --lr 0.001 \
#       --batch_size 100 --kappa_epoch 40 --eps_epoch 60 --eps_val 0.9 --eps_max 0.9 \
#       --reg_coef 10 | tee ${OUTDIR}/Adam_id.log
