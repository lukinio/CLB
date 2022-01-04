for fisherscale in 1e-10 1e-8 1e-5 1e-2; do
    for eps in 1e-10 1e-20 1e-30 1e-40; do
        for seed in 2001 2002 2003 2004 2005; do
            python train.py cfg=mnist cfg.normalize_shift=True cfg.seed=$seed cfg.eps=$eps cfg.epochs=10 cfg.robust_loss_threshold=$fisherscale tags=["20211228_fixed_fisher"] &
        done
        wait
    done
done
