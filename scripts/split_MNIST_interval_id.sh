GPUID=$1
OUTDIR=outputs/split_MNIST_interval_id_new
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

#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --wandb_logger --exp_tag ID_test --norm 2\
#       --eps_val 25000 10000 8000 5000 2000 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 4 --schedule 12 --kappa_min 0.5 --gradient_clipping 1 \
#       | tee ${OUTDIR}/experimental11.log


#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.0001 --clipping \
#       --wandb_logger --exp_tag ID_test_const_eps --norm 2 --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 \
#       --eps_val 25000 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 4 --schedule 12 --kappa_min 0.5 --gradient_clipping 1 \
#       | tee ${OUTDIR}/experimental11.log


#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --wandb_logger --exp_tag ID_eps_400000_wo_norm_wo_CI --norm 0 \
#       --eps_val 400000 --eps_epoch 20 --eps_max 400000 \
#       --kappa_epoch 20 --schedule 40 --kappa_min 0.5 --gradient_clipping 1 \
#       | tee ${OUTDIR}/experimental_wo_norm_wo_CI_eps#400000.log
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping \
#       --wandb_logger --exp_tag ID_eps_400000_norm_2_wo_CI --norm 2 --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 \
#       --eps_val 400000 --eps_epoch 20 --eps_max 400000 \
#       --kappa_epoch 20 --schedule 40 --kappa_min 0.5 --gradient_clipping 1 \
#       | tee ${OUTDIR}/experimental_norm_2_wo_CI_eps#400000.log
#
#python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 128 --lr 0.001 --clipping --clip_interval \
#       --wandb_logger --exp_tag ID_eps_400000_norm_2_CI --norm 2 --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 \
#       --eps_val 400000 --eps_epoch 20 --eps_max 400000 \
#       --kappa_epoch 20 --schedule 40 --kappa_min 0.5 --gradient_clipping 1 \
#       | tee ${OUTDIR}/experimental_norm_2_CI_eps#400000.log
#


python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.0001 --clipping \
       --wandb_logger --exp_tag ID_test_wo_norm_wo_CI --norm 0 \
       --eps_val 25000 10000 8000 5000 2000 --eps_epoch 12 --eps_max 0 \
       --kappa_epoch 4 --schedule 12 --kappa_min 0.5 --gradient_clipping 1 \
       | tee ${OUTDIR}/experimental1.log

python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.0001 --clipping \
       --wandb_logger --exp_tag ID_test_norm_2_wo_CI --norm 2 --reg_coef 0.0001 0.001 0.01 1 10 100\
       --eps_val 25000 10000 8000 5000 2000 --eps_epoch 12 --eps_max 0 \
       --kappa_epoch 4 --schedule 12 --kappa_min 0.5 --gradient_clipping 1 \
       | tee ${OUTDIR}/experimental2.log

python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --optimizer Adam --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.0001 --clipping --clip_interval \
       --wandb_logger --exp_tag ID_test_norm_2_CI --norm 2 --reg_coef 0.0001 0.001 0.01 1 10 100\
       --eps_val 25000 10000 8000 5000 2000 --eps_epoch 12 --eps_max 0 \
       --kappa_epoch 4 --schedule 12 --kappa_min 0.5 --gradient_clipping 1 \
       | tee ${OUTDIR}/experimental3.log