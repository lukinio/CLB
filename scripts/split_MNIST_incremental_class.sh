GPUID=$1
OUTDIR=outputs/split_MNIST_incremental_class
REPEAT=10
mkdir -p $OUTDIR
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400                                              --lr 0.001 --offline_training  | tee ${OUTDIR}/Offline.log
#CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400                                              --lr 0.001                     | tee ${OUTDIR}/Adam.log
#CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer SGD     --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400                                              --lr 0.01                      | tee ${OUTDIR}/SGD.log
#CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adagrad --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400                                              --lr 0.01                      | tee ${OUTDIR}/Adagrad.log
#CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name EWC_mnist        --lr 0.001 --reg_coef 600      | tee ${OUTDIR}/EWC.log
#CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name EWC_online_mnist --lr 0.001 --reg_coef 100      | tee ${OUTDIR}/EWC_online.log
#CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type regularization --agent_name SI  --lr 0.001 --reg_coef 600      | tee ${OUTDIR}/SI.log
#CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type regularization --agent_name L2  --lr 0.001 --reg_coef 100      | tee ${OUTDIR}/L2.log
#CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name Naive_Rehearsal_1100  --lr 0.001   | tee ${OUTDIR}/Naive_Rehearsal_1100.log
#CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name Naive_Rehearsal_4400  --lr 0.001   | tee ${OUTDIR}/Naive_Rehearsal_4400.log
#CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type regularization --agent_name MAS --lr 0.001 --reg_coef 1        | tee  ${OUTDIR}/MAS.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer SGD     --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name GEM_1100  --lr 0.01 --reg_coef 0.5 | tee  ${OUTDIR}/GEM_1100.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer SGD     --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name GEM_4400  --lr 0.01 --reg_coef 0.5 | tee  ${OUTDIR}/GEM_4400.log