ns=True
elr=10.
mr=1.
lam=100.
si=5
lr=0.001

idx=3
for seed in 2001 2002 2003 2004 2005; do
  for lam in 1; do
    for elr in 10; do
      export CUDA_VISIBLE_DEVICES=$idx
      python train.py cfg=id cfg.dataset=FASHION_MNIST cfg.seed=$seed cfg.optimizer=SGD cfg.interval.robust_accuracy_threshold=0.8 cfg.learning_rate=$lr cfg.interval.scale_init=$si cfg.interval.expansion_learning_rate=$elr cfg.interval.normalize_shift=$ns cfg.interval.normalize_scale=False cfg.interval.max_radius=$mr cfg.interval.robust_lambda=$lam cfg.epochs=30 tags=["20220124_fmnist_incremental_domain_semifinal"] &
      idx=$((idx + 1))
      if [ $idx -eq 4 ]; then
        idx=0
      fi
    done
  done
  wait
done
