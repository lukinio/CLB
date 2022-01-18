ns=True
elr=1.
mr=1.
lam=1000.
si=5
lr=0.001
elr=0.001
rat=0.8

# TODO
for lam in 1000 500; do
    for elr in 0.001 0.002; do
        for seed in 2001 2002 2003 2004 2005; do
            python train.py cfg=incremental_class cfg.seed=$seed cfg.optimizer=SGD cfg.interval.robust_accuracy_threshold=$rat cfg.learning_rate=$lr cfg.interval.scale_init=$si cfg.interval.expansion_learning_rate=$elr cfg.interval.normalize_shift=$ns cfg.interval.normalize_scale=False cfg.interval.max_radius=$mr cfg.interval.robust_lambda=$lam cfg.epochs=40 tags=["20220117_incremental_class_final_mnist"] &
        done
    done
    wait
done
