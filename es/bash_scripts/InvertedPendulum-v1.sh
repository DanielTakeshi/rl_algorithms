#!/bin/bash
clear
python main.py InvertedPendulum-v1 \
    --es_iters 800 \
    --lrate_es 0.005 \
    --log_every_t_iter 2 \
    --npop 200 \
    --seed 4 \
    --sigma 0.1 \
    --snapshot_every_t_iter 50 \
    --test_trajs 10 \
    --verbose
