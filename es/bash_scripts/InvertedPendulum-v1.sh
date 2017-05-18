#!/bin/bash
clear
python main.py InvertedPendulum-v1 \
    --es_iters 700 \
    --lrate_es 0.005 \
    --log_every_t_iter 2 \
    --npop 200 \
    --seed 5 \
    --sigma 0.1 \
    --snapshot_every_t_iter 50 \
    --test_trajs 10 \
    --verbose
