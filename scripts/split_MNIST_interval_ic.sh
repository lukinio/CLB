GPUID=$1
OUTDIR=outputs/split_MNIST_interval_ic_new
REPEAT=5
mkdir -p $OUTDIR

python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid $GPUID --repeat $REPEAT --incremental_class \
       --optimizer Adam --force_out_dim 10 --no_class_remap --first_split_size 2 \
       --other_split_size 2 --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.001 --clipping \
       --wandb_logger --exp_tag IC_test_wo_norm_wo_CI --norm 0 \
       --eps_val 25000 8000 4000 2000 1000 --eps_epoch 12 --eps_max 0 \
       --kappa_epoch 4 --schedule 12 --kappa_min 0.5 --gradient_clipping 1 \
       | tee ${OUTDIR}/experimental1.log

python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid $GPUID --repeat $REPEAT --incremental_class \
       --optimizer Adam --force_out_dim 10 --no_class_remap --first_split_size 2 \
       --other_split_size 2 --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.001 --clipping \
       --wandb_logger --exp_tag IC_test_norm_2_wo_CI --norm 2 --reg_coef 0.000001 0.00001 0.0001 0.001 \
       --eps_val 25000 8000 4000 2000 1000 --eps_epoch 12 --eps_max 0 \
       --kappa_epoch 4 --schedule 12 --kappa_min 0.5 --gradient_clipping 1 \
       | tee ${OUTDIR}/experimental2.log

python -u intervalBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid $GPUID --repeat $REPEAT --incremental_class \
       --optimizer Adam --force_out_dim 10 --no_class_remap --first_split_size 2 \
       --other_split_size 2 --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 128 --lr 0.001 --clipping --clip_interval \
       --wandb_logger --exp_tag IC_test_norm_2_CI --norm 2 --reg_coef 0.000001 0.00001 0.0001 0.001 \
       --eps_val 25000 8000 4000 2000 1000 --eps_epoch 12 --eps_max 0 \
       --kappa_epoch 4 --schedule 12 --kappa_min 0.5 --gradient_clipping 1 \
       | tee ${OUTDIR}/experimental3.log