#!/bin/bash
rm -r outputs/HalfCheetah-v1/seed0001
clear
python main.py HalfCheetah-v1 \
    --es_iters 1000 \
    --log_every_t_iter 1 \
    --npop 200 \
    --seed 1 \
    --sigma 0.1 \
    --test_trajs 10 \
    --verbose
