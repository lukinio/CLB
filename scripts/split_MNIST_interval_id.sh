GPUID=$1
OUTDIR=outputs/split_MNIST_interval_id
REPEAT=5
mkdir -p $OUTDIR


#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 25000 10000 8000 5000 2000 --eps_epoch 10 --eps_max 0 \
#       --kappa_epoch 2 --schedule 10 --kappa_min 0.5 \
#       | tee ${OUTDIR}/experimental.log

#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 25000 10000 8000 5000 2000 --eps_epoch 10 --eps_max 0 \
#       --kappa_epoch 2 --schedule 10 --kappa_min 0.5 \
#       | tee ${OUTDIR}/experimental1.log

#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 25000 10000 8000 5000 2000 --eps_epoch 10 --eps_max 0 \
#       --kappa_epoch 2 --schedule 10 --kappa_min 0.5 \
#       | tee ${OUTDIR}/experimental2.log

#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 25000 10000 8000 5000 2000 --eps_epoch 12 --eps_max 25000 10000 8000 5000 2000 \
#       --kappa_epoch 4 --schedule 12 --kappa_min 0.5 \
#       | tee ${OUTDIR}/experimental3.log

#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 50000 20000 16000 10000 3000 --eps_epoch 24 --eps_max 0 \
#       --kappa_epoch 8 --schedule 24 --kappa_min 0.5 \
#       | tee ${OUTDIR}/experimental4.log

#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 25000 10000 8000 5000 2000 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 8 --schedule 12 --kappa_min 0.5 \
#       | tee ${OUTDIR}/experimental5.log

#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 25000 10000 8000 5000 2000 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 4 --schedule 12 --kappa_min 0.5 \
#       | tee ${OUTDIR}/experimental6.log


#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 25000 10000 8000 5000 2000 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 4 --schedule 12 --kappa_min 0.5 --weight_decay 3e-4 \
#       | tee ${OUTDIR}/experimental7.log

#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 25000 10000 8000 5000 2000 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 4 --schedule 12 --kappa_min 0.5 --gradient_clipping 1 \
#       | tee ${OUTDIR}/experimental8.log

#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat 10 \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 25000 10000 8000 5000 2000 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 4 --schedule 12 --kappa_min 0.5 --gradient_clipping 1 \
#       | tee ${OUTDIR}/experimental9.log

python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.0001 --clipping \
       --eps_val 25000 10000 8000 5000 2000 --eps_epoch 12 --eps_max 0 \
       --kappa_epoch 0 2 --warm_epoch 3 --schedule 12 --kappa_min 1 0 --gradient_clipping 1 \
       | tee ${OUTDIR}/experimental11.log

#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 40000 20000 10000 5000 2500 --eps_epoch 1 --eps_max 0 \
#       --kappa_epoch 1 --schedule 1 --kappa_min 0.5 \
#       | tee ${OUTDIR}/experimental4.log

