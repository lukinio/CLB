GPUID=$1
OUTDIR=outputs/split_MNIST_interval_it_per_weight
REPEAT=10
mkdir -p $OUTDIR

python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
       --force_out_dim 0 --first_split_size 2 --other_split_size 2 --model_name MLP400 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
       --eps_val 14 9 7 5 3 --eps_epoch 12 --eps_max 0 \
       --kappa_epoch 1 --schedule 12 \
       | tee ${OUTDIR}/experimental.log