GPUID=$1
OUTDIR=outputs/split_MNIST_interval_id
REPEAT=2
mkdir -p $OUTDIR

python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.001 --clipping --kl \
       --exp_tag ID_softmax_KL_good \
       --eps_val 1140800 --eps_mode sum --eps_actv_mode softmax \
       --gradient_clipping 1 --reg_coef 0.001 0.01 0.1 10 100 \
       | tee ${OUTDIR}/interval_id_KL1_good.log


python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.001 --clipping --norm max \
       --exp_tag ID_softmax_norm_max_good \
       --eps_val 1140800 --eps_mode sum --eps_actv_mode softmax \
       --gradient_clipping 1 --reg_coef 0.001 0.01 0.1 10 100 \
       | tee ${OUTDIR}/interval_id_norm_max_good.log