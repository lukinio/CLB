GPUID=$1
OUTDIR=outputs/split_CIFAR100_cnn_base
REPEAT=5
mkdir -p $OUTDIR

# =========================== Incremental task ========================================
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn                                              --lr 0.001                            | tee ${OUTDIR}/Adam_it.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer SGD     --force_out_dim 0 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn                                              --lr 0.01                             | tee ${OUTDIR}/SGD_it.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adagrad --force_out_dim 0 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn                                              --lr 0.01                             | tee ${OUTDIR}/Adagrad_it.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type customization  --agent_name EWC_online --lr 0.001 --reg_coef 3000     | tee ${OUTDIR}/EWC_online_it.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type customization  --agent_name EWC        --lr 0.001 --reg_coef 100      | tee ${OUTDIR}/EWC_it.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type regularization --agent_name SI  --lr 0.001 --reg_coef 2               | tee ${OUTDIR}/SI_it.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type regularization --agent_name L2  --lr 0.001 --reg_coef 1               | tee ${OUTDIR}/L2_it.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type regularization --agent_name MAS --lr 0.001 --reg_coef 10              | tee ${OUTDIR}/MAS_it.log



# =========================== Incremental domain =========================================
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 10 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn                                              --lr 0.001                            | tee ${OUTDIR}/Adam_id.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer SGD     --force_out_dim 10 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn                                              --lr 0.01                             | tee ${OUTDIR}/SGD_id.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adagrad --force_out_dim 10 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn                                              --lr 0.01                             | tee ${OUTDIR}/Adagrad_id.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 10 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type customization  --agent_name EWC_online --lr 0.001 --reg_coef 20       | tee ${OUTDIR}/EWC_online_id.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 10 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type customization  --agent_name EWC        --lr 0.001 --reg_coef 10       | tee ${OUTDIR}/EWC_id.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 10 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type regularization --agent_name SI         --lr 0.001 --reg_coef 10000    | tee ${OUTDIR}/SI_id.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 10 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type regularization --agent_name L2         --lr 0.001 --reg_coef 0.0001   | tee ${OUTDIR}/L2_id.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 10 --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type regularization --agent_name MAS        --lr 0.001 --reg_coef 1000000  | tee ${OUTDIR}/MAS_id.log


# =========================== Incremental class =========================================
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 100 --no_class_remap --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn                                             --lr 0.001                                 | tee ${OUTDIR}/Adam_ic.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer SGD     --force_out_dim 100 --no_class_remap --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn                                             --lr 0.1                                   | tee ${OUTDIR}/SGD_ic.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adagrad --force_out_dim 100 --no_class_remap --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn                                             --lr 0.1                                   | tee ${OUTDIR}/Adagrad_ic.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 100 --no_class_remap --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type customization  --agent_name EWC        --lr 0.001 --reg_coef 2            | tee ${OUTDIR}/EWC_ic.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 100 --no_class_remap --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type customization  --agent_name EWC_online --lr 0.001 --reg_coef 2            | tee ${OUTDIR}/EWC_online_ic.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 100 --no_class_remap --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type regularization --agent_name SI         --lr 0.001 --reg_coef 0.001        | tee ${OUTDIR}/SI_ic.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 100 --no_class_remap --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type regularization --agent_name L2         --lr 0.001 --reg_coef 500          | tee ${OUTDIR}/L2_ic.log
CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 100 --no_class_remap --first_split_size 10 --other_split_size 10 --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type regularization --agent_name MAS        --lr 0.001 --reg_coef 0.001        | tee ${OUTDIR}/MAS_ic.log




#CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --optimizer Adam --force_out_dim 0 --first_split_size 10 --other_split_size 10 \
#       --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type customization \
#        --agent_name EWC --lr 0.001 --reg_coef 100 | tee ${OUTDIR}/EWC_it_max.log
#
#
#CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --optimizer Adam --force_out_dim 10 --first_split_size 10 --other_split_size 10 \
#       --schedule 100 --batch_size 128 --model_name cnn --model_type cnn --agent_type customization \
#       --agent_name EWC --lr 0.001 --reg_coef 10 | tee ${OUTDIR}/EWC_id_max.log


#CUDA_VISIBLE_DEVICES=$GPUID python -u iBatchLearn.py --dataset CIFAR100 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --incremental_class --optimizer Adam --force_out_dim 100 --no_class_remap \
#       --first_split_size 10 --other_split_size 10 --model_name cnn --model_type cnn \
#       --agent_type customization --agent_name EWC --lr 0.001 --batch_size 128 \
#       --schedule 100 | tee ${OUTDIR}/EWC_ic_max.log

