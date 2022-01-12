ns=True
elr=10.
mr=1.
lam=100.
si=5
lr=1.

for elr in 10; do
    for seed in 2001; do
        python train.py cfg=incremental_domain cfg.seed=$seed cfg.optimizer=SGD cfg.interval.robust_accuracy_threshold=0.8 cfg.learning_rate=$lr cfg.interval.scale_init=$si cfg.interval.expansion_learning_rate=$elr cfg.interval.normalize_shift=$ns cfg.interval.normalize_scale=False cfg.interval.max_radius=$mr cfg.interval.robust_lambda=$lam cfg.epochs=30
    done
    wait
done
