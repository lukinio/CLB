GPUID=$1
OUTDIR=outputs/split_MNIST_interval_id_per_weight
REPEAT=5
mkdir -p $OUTDIR

#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 12 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --kappa_epoch 4 --eps_epoch 12 --eps_val 10 3.3 1.1 0.36 0.12 \
#       --eps_max 0 --clipping | tee ${OUTDIR}/in_pw_random.log

#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 10 --eps_epoch 8 --eps_max 0 \
#       --kappa_epoch 4 --schedule 8 \
#       | tee ${OUTDIR}/in_pw_random2.log

#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 6 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 4 --schedule 12 \
#       | tee ${OUTDIR}/in_pw_random3.log

#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 16 4 3 1 0.5 --eps_epoch 12 10 10 10 8 --eps_max 0 \
#       --kappa_epoch 1 --schedule 12 10 10 10 8 \
#       | tee ${OUTDIR}/in_pw_random4.log


#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 16 8 8 3 2 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 1 --schedule 12 \
#       | tee ${OUTDIR}/in_pw.log

# --eps_val 20 8 4 2 1
#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 18 8 4 1 0.5 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 1 --schedule 12 \
#       | tee ${OUTDIR}/in_pwd.log

#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 18 8 4 1 0.5 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 1 --schedule 12 \
#       | tee ${OUTDIR}/in_pwd1.log

#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 60 40 20 10 5 --eps_epoch 20 --eps_max 0 \
#       --kappa_epoch 1 --schedule 20 \
#       | tee ${OUTDIR}/in_pwd2.log


#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 18 8 4 1 0.5 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 1 --schedule 12 \
#       | tee ${OUTDIR}/in_pw1.log
#

#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 20 8 5 3 1 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 1 --schedule 12 \
#       | tee ${OUTDIR}/in_pw1_test.log

#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 50 20 10 5 2 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 1 --schedule 12 --kappa_min 0.5 --weight_decay 5e-1 --reg_coef 200 \
#       | tee ${OUTDIR}/in_pw1_test1.log


#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 50 20 10 5 2 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 1 --schedule 12 --kappa_min 0.5 --weight_decay 5e-1 --reg_coef 200 \
#       | tee ${OUTDIR}/in_pw1_test2.log
#
#
#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 50 20 10 5 2 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 1 --schedule 12 --kappa_min 0.5 \
#       | tee ${OUTDIR}/in_pw1_test3.log
#
#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 50 20 10 5 2 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 1 --schedule 12 --kappa_min 0.5 --weight_decay 5e-1 \
#       | tee ${OUTDIR}/in_pw1_test4.log
#
#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 50 20 10 5 2 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 1 --schedule 12 --kappa_min 0.5 --reg_coef 200 \
#       | tee ${OUTDIR}/in_pw1_test5.log




#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 150 50 20 15 12 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 1 --schedule 12 --kappa_min 0.5 --weight_decay 5e-3 \
#       | tee ${OUTDIR}/in_pw1_test4.log

#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 20 10 4 4 3 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 1 --schedule 12 --kappa_min 0.5  \
#       | tee ${OUTDIR}/in_pw1_test5.log

#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 20 10 4 4 3 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 1 2 3 4 5 --schedule 12 --kappa_min 0  \
#       | tee ${OUTDIR}/in_pw1_test6.log
#
#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 20 10 4 4 3 --eps_epoch 12 --eps_max 0 \
#       --kappa_epoch 1 --schedule 12 --kappa_min 0  \
#       | tee ${OUTDIR}/in_pw1_test7.log


#python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
#       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
#       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
#       --eps_val 25 15 9 9 7 --eps_epoch 6 --eps_max 20 10 4 4 3 \
#       --kappa_epoch 6 --schedule 20 --kappa_min 0 \
#       | tee ${OUTDIR}/in_pw1_test8.log


python -u intervalBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --optimizer Adam \
       --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet \
       --batch_size 100 --lr 0.001 --clipping --eps_per_model \
       --eps_val 25 15 9 9 7 --eps_epoch 6 --eps_max 20 10 4 4 3 \
       --kappa_epoch 6 --schedule 20 --kappa_min 0 \
       | tee ${OUTDIR}/in_pw1_test9.log











# --weight_decay 5e-3




# 16 4 3 1 0.5 - 80%
# 16 4 3 1 1
# --eps_val 15 4 3 1 1 81% - z exp.sum(dim=1)[:, None] BT i --eps_per_model
# --eps_val 100 10 1 0.1 0.01 acc 75% - z exp.sum(dim=1)[:, None] BT
# --eps_val 6000 2000 666 222 111
