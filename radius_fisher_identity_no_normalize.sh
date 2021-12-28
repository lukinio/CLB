for fisherscale in 1e-5 1e-3 1e-2 1e-1 1 1e1 1e2; do
    for seed in 2001 2002 2003 2004 2005; do
        python train.py cfg=mnist cfg.normalize_shift=False cfg.seed=$seed cfg.eps=1. cfg.fisher_mode=identity cfg.epochs=5 cfg.robust_loss_threshold=$fisherscale tags=["20211227_identity_nonormalize"] &
    done
    wait
done
