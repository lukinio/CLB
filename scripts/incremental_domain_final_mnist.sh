ns=True
elr=10.
mr=1.
lam=1.
si=5
lr=0.001

for lam in 100; do
    for lr in 1; do
        for rat in 0.5 0.9; do
            for seed in 2001 2002 2003 2004 2005; do
                    python train.py cfg=incremental_domain cfg.seed=$seed cfg.optimizer=SGD cfg.interval.robust_accuracy_threshold=$rat cfg.learning_rate=$lr cfg.interval.scale_init=$si cfg.interval.expansion_learning_rate=$elr cfg.interval.normalize_shift=$ns cfg.interval.normalize_scale=False cfg.interval.max_radius=$mr cfg.interval.robust_lambda=$lam cfg.epochs=40 tags=["20220117_incremental_domain_40ep"] &
            done
        done
        wait
    done
done
