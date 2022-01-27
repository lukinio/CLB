ns=True
elr=0.001
mr=1.
lam=1000.
si=5
lr=0.001


for seed in 2001 2002 2003 2004 2005; do 
    for rat in 0.8; do
        for lam in 1000; do
            python train.py cfg=it cfg.scenario=INC_CLASS cfg.interval.max_radius=${mr} cfg.interval.robust_accuracy_threshold=$rat cfg.interval.robust_lambda=$lam cfg.interval.expansion_learning_rate=$elr cfg.learning_rate=$lr  cfg.interval.scale_init=5.0
        done
        wait
    done
done
