ns=True
elr=1.
mr=1.
lam=100.
si=5
lr=1.

for ns in True; do
    for seed in 2001 2002 2003 2004 2005; do
        python train.py cfg=incremental_task cfg.optimizer=SGD cfg.seed=$seed cfg.learning_rate=$lr cfg.interval.scale_init=$si cfg.interval.expansion_learning_rate=$elr cfg.interval.normalize_shift=$ns cfg.interval.normalize_scale=False cfg.interval.max_radius=$mr cfg.interval.robust_lambda=$lam cfg.epochs=30 tags=["20220111_incremental_task_semifinal"] &
    done
done
wait
