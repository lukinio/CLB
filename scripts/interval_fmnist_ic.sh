ns=True
elr=1.
mr=1.
lam=1000.
si=5
lr=0.001
rat=0.9

for seed in 2001 2002 2003 2004 2005; do
  for elr in 0.001; do
    for lam in 100; do
      export CUDA_VISIBLE_DEVICES=$idx
      python train.py cfg=it cfg.dataset=FASHION_MNIST cfg.seed=$seed cfg.scenario=INC_CLASS cfg.optimizer=SGD cfg.interval.max_radius=${mr} cfg.interval.robust_accuracy_threshold=$rat cfg.interval.robust_lambda=$lam cfg.interval.expansion_learning_rate=$elr cfg.learning_rate=$lr cfg.interval.scale_init=5.0 tags=["20220124_fmnist_incremental_class_semifinal"] &
      idx=$((idx + 1))
      if [ $idx -eq 2 ]; then
        idx=0
      fi
    done
  done
  wait
done
