for seed in 2001 2002 2003 2004 2005; do
    python train.py tags=["20211228_checking_random_seeds"] cfg.seed=$seed &
done
