GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_base
REPEAT=10
mkdir -p $OUTDIR


CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
       --repeat $REPEAT --optimizer Adam --force_out_dim 0 --first_split_size 2 \
       --other_split_size 2 --schedule 50 --batch_size 100 --model_name cnn \
       --model_type cnn --agent_type customization --agent_name EWC --lr 0.001 \
       --reg_coef 100 | tee ${OUTDIR}/EWC_it_max.log


CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID \
       --repeat $REPEAT --optimizer Adam --force_out_dim 2 --first_split_size 2 \
       --other_split_size 2 --schedule 50 --batch_size 100 --model_name cnn \
       --model_type cnn --agent_type customization --agent_name EWC --lr 0.001 \
       --reg_coef 10 | tee ${OUTDIR}/EWC_id_max.log
