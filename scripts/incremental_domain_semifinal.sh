ns=True
elr=10.
mr=1.
lam=100.
si=5
lr=1.

python train.py cfg=incremental_domain cfg.optimizer=SGD cfg.learning_rate=$lr cfg.interval.scale_init=$si cfg.interval.expansion_learning_rate=$elr cfg.interval.normalize_shift=$ns cfg.interval.normalize_scale=False cfg.interval.max_radius=$mr cfg.interval.robust_lambda=$lam cfg.epochs=30
