for rbt in 1 5 10 20 50 75 100 500 1000; do
    for seed in 2001 2002 2003 2004 2005; do
        python train.py cfg=mnist2 cfg.seed=$seed cfg.robust_loss_threshold=$rbt tags=["robust_threshold_grid"] &
    done
    wait
done
