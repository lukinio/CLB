GPUID=$1
OUTDIR=outputs/split_MNIST_interval_id
REPEAT=1
mkdir -p $OUTDIR


#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 1000 --eps_epoch 5 --eps_max 0 \
#       --kappa_epoch 0 2 --warm_epoch 3 --schedule 20 --kappa_min 1 0 --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental11.log

#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 30000 --eps_epoch 24 --eps_max 0 \
#       --kappa_epoch 0 4 --warm_epoch 5 --schedule 24 --kappa_min 1 0 --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental12.log


#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 80000 --eps_epoch 35 --eps_max 0 \
#       --kappa_epoch 0 6 --warm_epoch 7 --schedule 35 --kappa_min 1 0 --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental13.log

#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 55000 --eps_epoch 50 --eps_max 0 \
#       --kappa_epoch 0 6 --warm_epoch 7 --schedule 50 --kappa_min 1 0 --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental14.log
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 110000 --eps_epoch 100 --eps_max 0 \
#       --kappa_epoch 0 7 --warm_epoch 8 --schedule 100 --kappa_min 1 0 --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental15.log


#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 40000 --eps_epoch 50 --eps_max 0 --milestones 2 \
#       --kappa_epoch 0 4 --warm_epoch 4 --schedule 50 --kappa_min 1 0 --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental16.log

#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 40000 --eps_epoch 100 --eps_max 0 --milestones 2 \
#       --kappa_epoch 0 4 --warm_epoch 4 --schedule 100 --kappa_min 1 0 --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental17.log
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 40000 --eps_epoch 90 --eps_max 0 --milestones 2 \
#       --kappa_epoch 0 10 --warm_epoch 10 --schedule 90 --kappa_min 1 0 --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental18.log
#
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 40000 --eps_epoch 100 --eps_max 0 --milestones 2 \
#       --kappa_epoch 0 10 --warm_epoch 10 --schedule 100 --kappa_min 1 0 --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental19.log


python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.0001 --clipping \
       --eps_val 40000 --eps_epoch 100 --eps_max 0 --milestones 2 \
       --kappa_epoch 0 10 --warm_epoch 10 --schedule 100 --kappa_min 1 0 --gradient_clipping 1 \
       | tee ${OUTDIR}/dis_experimental21.log