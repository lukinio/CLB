GPUID=$1
OUTDIR=outputs/split_CIFAR10_cnn_interval_id
REPEAT=2
mkdir -p $OUTDIR

python -u intervalBatchLearn.py --dataset CIFAR10 --train_aug --gpuid "${GPUID}" \
       --repeat "${REPEAT}" --optimizer Adam --force_out_dim 2 --first_split_size 2 \
       --other_split_size 2 --model_name interval_cnn --model_type cnn --agent_type interval \
       --agent_name IntervalNet --batch_size 128 --lr 0.001 --clipping \
       --init_eps 0.001 --wc_threshold 0.8 \
       --kappa_epoch 20 --schedule 40 --kappa_min 0.0 \
       | tee ${OUTDIR}/experimental.log