GPUID=$1
OUTDIR=outputs/split_MNIST_interval_id
REPEAT=2
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


#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 40000 --eps_epoch 100 --eps_max 0 --milestones 2 \
#       --kappa_epoch 0 10 --warm_epoch 10 --schedule 100 --kappa_min 1 0 --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental21.log

#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --eps_val 40000 --eps_epoch 100 --eps_max 0 --milestones 2 \
#       --kappa_epoch 0 10 --warm_epoch 10 --schedule 100 --kappa_min 1 0 --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental22.log


#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 40000 --eps_epoch 100 --eps_max 0 --milestones 2 \
#       --kappa_epoch 0 10 --warm_epoch 10 --schedule 100 --kappa_min 1 0 --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental24.log



#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1e+12 --eps_mode product \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental25.log
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 100 --eps_mode product \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental26.log
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1e+6 --eps_mode product \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental27.log


#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 114080 --eps_mode sum --eps_actv_mode softmax \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental28_softmax_fast.log

# ===========================
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1140800 --eps_mode sum --eps_actv_mode softmax \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental28_softmax.log
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1140800 --eps_mode sum --eps_actv_mode other \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental28_other.log
#
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 114080 --eps_mode sum --eps_actv_mode softmax \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental29_softmax.log
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 114080 --eps_mode sum --eps_actv_mode other \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental29_other.log
#
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1e+12 --eps_mode product --eps_actv_mode softmax \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental25_softmax.log
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1e+12 --eps_mode product --eps_actv_mode other \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental25_other.log


#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1140800 --eps_mode sum --eps_actv_mode softmax \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental30_softmax.log
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1140800 --eps_mode sum --eps_actv_mode other \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental30_other.log
#
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 114080 --eps_mode sum --eps_actv_mode softmax \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental31_softmax.log
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 114080 --eps_mode sum --eps_actv_mode other \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental31_other.log
#
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1e+12 --eps_mode product --eps_actv_mode softmax \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental32_softmax.log
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1e+12 --eps_mode product --eps_actv_mode other \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental32_other.log


#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1140800 --eps_mode sum --eps_actv_mode softmax \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental33_softmax.log
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1140800 --eps_mode sum --eps_actv_mode other \
#       --gradient_clipping 1 \
#       | tee ${OUTDIR}/dis_experimental33_other.log


#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1140800 --eps_mode sum --eps_actv_mode softmax \
#       --gradient_clipping 1 --reg_coef 10 \
#       | tee ${OUTDIR}/dis_experimental34_softmax.log



#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1140800 --eps_mode sum --eps_actv_mode softmax \
#       --gradient_clipping 1 --reg_coef 0.1 \
#       | tee ${OUTDIR}/dis_experimental35_softmax.log
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --eps_val 1140800 --eps_mode sum --eps_actv_mode other \
#       --gradient_clipping 1 --reg_coef 10 0.1 \
#       | tee ${OUTDIR}/dis_experimental345_other.log



python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.001 --clipping --kl \
       --exp_tag ID_softmax_KL \
       --eps_val 1140800 --eps_mode sum --eps_actv_mode softmax \
       --gradient_clipping 1 --reg_coef 0.01 0.1 10 100 \
       | tee ${OUTDIR}/interval_id_KL.log


python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.001 --clipping --norm 2 \
       --exp_tag ID_softmax_norm2 \
       --eps_val 1140800 --eps_mode sum --eps_actv_mode softmax \
       --gradient_clipping 1 --reg_coef 0.01 0.1 10 100 \
       | tee ${OUTDIR}/interval_id_norm2.log


python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.001 --clipping --norm max \
       --exp_tag ID_softmax_norm_max \
       --eps_val 1140800 --eps_mode sum --eps_actv_mode softmax \
       --gradient_clipping 1 --reg_coef 0.01 0.1 10 100 \
       | tee ${OUTDIR}/interval_id_norm_max.log


python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.001 --clipping --norm 1 \
       --exp_tag ID_softmax_norm1 \
       --eps_val 1140800 --eps_mode sum --eps_actv_mode softmax \
       --gradient_clipping 1 --reg_coef 0.01 0.1 10 100 \
       | tee ${OUTDIR}/interval_id_norm1.log
