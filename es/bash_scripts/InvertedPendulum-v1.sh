#!/bin/bash
rm -r outputs/InvertedPendulum-v1/seed0002
clear
python main.py InvertedPendulum-v1 \
    --es_iters 500 \
    --lrate_es 0.002 \
    --log_every_t_iter 1 \
    --npop 50 \
    --seed 2 \
    --sigma 0.05 \
    --snapshot_every_t_iter 25 \
    --test_trajs 10 \
    --verbose
