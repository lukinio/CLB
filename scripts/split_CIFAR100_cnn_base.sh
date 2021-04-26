GPUID=$1
OUTDIR=outputs/split_CIFAR100_cnn_base1
REPEAT=5
mkdir -p $OUTDIR

CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT \
       --optimizer Adam --force_out_dim 10 --first_split_size 10 --other_split_size 10 \
       --schedule 100 --batch_size 128 --model_name cnn_avg --model_type cnn --agent_type customization \
       --agent_name EWC --lr 0.001 --reg_coef 10 | tee ${OUTDIR}/EWC_id_avg.log


CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT \
       --optimizer Adam --force_out_dim 0 --first_split_size 10 --other_split_size 10 \
       --schedule 100 --batch_size 128 --model_name cnn_avg --model_type cnn --agent_type customization \
        --agent_name EWC --lr 0.001 --reg_coef 100 | tee ${OUTDIR}/EWC_it_avg.log



CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT \
       --optimizer Adam --force_out_dim 0 --first_split_size 10 --other_split_size 10 \
       --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type customization \
        --agent_name EWC --lr 0.001 --reg_coef 100 | tee ${OUTDIR}/EWC_it_max.log


CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT \
       --optimizer Adam --force_out_dim 10 --first_split_size 10 --other_split_size 10 \
       --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type customization \
       --agent_name EWC --lr 0.001 --reg_coef 10 | tee ${OUTDIR}/EWC_id_max.log
