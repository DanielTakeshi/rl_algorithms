#!/bin/bash
rm -r outputs/InvertedPendulum-v1/seed0003
clear
python main.py InvertedPendulum-v1 \
    --es_iters 1000 \
    --lrate_es 0.005 \
    --log_every_t_iter 1 \
    --npop 100 \
    --seed 3 \
    --sigma 0.1 \
    --snapshot_every_t_iter 50 \
    --test_trajs 10 \
    --verbose
