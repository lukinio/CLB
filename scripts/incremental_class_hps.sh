ns=True
elr=1.
mr=1.
lam=100.
si=5
lr=0.001
elr=0.01
rat=0.8

for lam in 100 50 20; do
    for elr in 0.01 0.02 0.05; do
        for rat in 0.1 0.2 0.8; do
            python train.py cfg=incremental_class cfg.optimizer=SGD cfg.interval.robust_accuracy_threshold=$rat cfg.learning_rate=$lr cfg.interval.scale_init=$si cfg.interval.expansion_learning_rate=$elr cfg.interval.normalize_shift=$ns cfg.interval.normalize_scale=False cfg.interval.max_radius=$mr cfg.interval.robust_lambda=$lam cfg.epochs=30 tags=["20220116_incremental_class_hps"] &
        done
    done
    wait
done
