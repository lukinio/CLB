GPUID=$1
OUTDIR=outputs/split_MNIST_interval_id_per_weight
REPEAT=5
mkdir -p $OUTDIR


python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.001 --clipping --eps_per_model \
       --eps_val 25000 --eps_epoch 6 --eps_max 25000 \
       --kappa_epoch 6 --schedule 20 --kappa_min 0 \
       | tee ${OUTDIR}/experimental.log