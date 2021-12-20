for fisherscale in 1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7; do
    for eps in 1e-10 1e-8 1e-6 1e-4 1e-2 1; do
        for seed in 2001 2002 2003 2004 2005; do
            python train.py cfg=mnist2 cfg.seed=$seed cfg.eps=$eps cfg.robust_loss_threshold=$fisherscale tags=["20211219_radius_fisher_mnist_longer_better"] &
        done
        wait
    done
done
