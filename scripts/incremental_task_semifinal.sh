ns=True
elr=1.
mr=1.
lam=100.
si=5
lr=0.001

for ns in True; do
    for elr in 0.001 0.01 0.1 1 10; do
        python train.py cfg=incremental_task cfg.optimizer=SGD cfg.learning_rate=$lr cfg.interval.scale_init=$si cfg.interval.expansion_learning_rate=$elr cfg.interval.normalize_shift=$ns cfg.interval.normalize_scale=False cfg.interval.max_radius=$mr cfg.interval.robust_lambda=$lam cfg.epochs=30 tags=["20220114_incremental_task_lam1_hps"] &
    done
done
