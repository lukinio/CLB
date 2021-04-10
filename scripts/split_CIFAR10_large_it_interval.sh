GPUID=$1
OUTDIR=outputs/split_CIFAR10_large_interval_it_test
REPEAT=1
mkdir -p $OUTDIR


#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 12 --model_name large --model_type large \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 4 --eps_epoch 12 --eps_val 2.5 1.1 0.5 0.2 0.1 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/Adam_it.log
#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 12 --model_name large --model_type large \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 4 --eps_epoch 12 --eps_val 10 3.3 1.1 0.36 0.12 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test1.log
#
#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
#       --other_split_size 2 --schedule 12 --model_name large --model_type large \
#       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
#       --kappa_epoch 4 --eps_epoch 12 --eps_val 2 0.9 0.4 0.1 0.1 --eps_max 0 \
#       --clipping | tee ${OUTDIR}/test2.log

python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
       --other_split_size 2 --schedule 12 --model_name large --model_type large \
       --agent_type interval --agent_name IntervalNet --lr 0.001 --batch_size 100 \
       --kappa_epoch 4 --eps_epoch 12 --eps_val 5 1.8 0.5 0.2 0.1 --eps_max 0 \
       --clipping | tee ${OUTDIR}/test3.log

#python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
#       --repeat $REPEAT --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#       --other_split_size 2 --schedule 60 --model_name interval_cnn \
#       --model_type cnn --agent_type interval --agent_name IntervalNet --lr 0.001 \
#       --batch_size 100 --kappa_epoch 40 --eps_epoch 60 --eps_val 0.9 --eps_max 0.9 \
#       --reg_coef 10 | tee ${OUTDIR}/Adam_id.log
