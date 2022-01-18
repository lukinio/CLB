ns=True
elr=10.
mr=1.
lam=100.
si=5
lr=1.
rat=0.8

for rat in 0.5 0.7 0.9; do
    for lam in 1 5; do
        for elr in 100. 50. 10; do
            python train.py cfg=incremental_domain cfg.optimizer=SGD cfg.interval.robust_accuracy_threshold=$rat cfg.learning_rate=$lr cfg.interval.scale_init=$si cfg.interval.expansion_learning_rate=$elr cfg.interval.normalize_shift=$ns cfg.interval.normalize_scale=False cfg.interval.max_radius=$mr cfg.interval.robust_lambda=$lam cfg.epochs=30 tags=["20220113_incremental_domain_hps"] &
        done
    done
    wait
done
