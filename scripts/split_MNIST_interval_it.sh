GPUID=$1
OUTDIR=outputs/split_MNIST_interval_it
REPEAT=10
mkdir -p $OUTDIR

python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 0 \
       --first_split_size 2 --other_split_size 2 --schedule 30 --batch_size 100 --model_name MLP400 \
       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet --lr 0.001 \
       --kappa_epoch 10 --eps_epoch 30 --eps_val 0.4 --eps_max 0.4 \
       | tee ${OUTDIR}/IN_Adam_tr_eps04.log



# Incremental Task

#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 0 \
#       --first_split_size 2 --other_split_size 2 --schedule 12 --batch_size 100 --model_name MLP400 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet --lr 0.001 \
#       --kappa_epoch 4 --eps_epoch 12 --eps_val 0.1 --eps_max 0.1 \
#       | tee ${OUTDIR}/IN_Adam_01_eps.log

#good
#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 0 \
#       --first_split_size 2 --other_split_size 2 --schedule 30 --batch_size 100 --model_name MLP400 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet --lr 0.001 \
#       --kappa_epoch 10 --eps_epoch 30 --eps_val 0.4 --eps_max 0.4 \
#       | tee ${OUTDIR}/IN_Adam_02_eps.log


#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 0 \
#       --first_split_size 2 --other_split_size 2 --schedule 30 --batch_size 100 --model_name MLP400 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet --lr 0.001 \
#       --kappa_epoch 10 --eps_epoch 30 --eps_val 0.1 --eps_max 0.1 \
#       | tee ${OUTDIR}/IN_Adam_03_eps.log









#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 0 \
#       --first_split_size 2 --other_split_size 2 --schedule 8 --batch_size 128 --model_name MLP400 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet --lr 0.001 \
#       --kappa_epoch 4 --eps_epoch 8 --eps_val 0.1 --eps_max 0.1 --clipping \
#       | tee ${OUTDIR}/IN_Adam_01_eps_clipping.log
#
#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 0 \
#       --first_split_size 2 --other_split_size 2 --schedule 16 --batch_size 128 --model_name MLP400 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet --lr 0.001 \
#       --kappa_epoch 8 --eps_epoch 16 --eps_val 0.2 --eps_max 0.2 \
#       | tee ${OUTDIR}/IN_Adam_02_eps.log
#
#python -u intervalBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 0 \
#       --first_split_size 2 --other_split_size 2 --schedule 16 --batch_size 128 --model_name MLP400 \
#       --model_name interval_mlp400 --agent_type interval --agent_name IntervalNet --lr 0.001 \
#       --kappa_epoch 8 --eps_epoch 16 --eps_val 0.2 --eps_max 0.2 --clipping \
#       | tee ${OUTDIR}/IN_Adam_02_eps_clipping.log
