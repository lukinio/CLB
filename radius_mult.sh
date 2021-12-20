for rad in 1000 10000; do
    python train.py cfg=two1dfunctions cfg.radius_multiplier=$rad tags=["20211214_searching_radius_mult_lowinit_noscale"] &
done
