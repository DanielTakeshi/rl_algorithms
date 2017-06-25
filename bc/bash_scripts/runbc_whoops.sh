#!/bin/bash
set -eux
for s in 0 1 2; do
    python bc.py Ant-v1         25 --test_rollouts 50 --eval_freq 100 --train_iters 20001 --seed $s --subsamp_freq 20
    python bc.py HalfCheetah-v1 25 --test_rollouts 50 --eval_freq 100 --train_iters 20001 --seed $s --subsamp_freq 20
    python bc.py Hopper-v1      25 --test_rollouts 50 --eval_freq 100 --train_iters 20001 --seed $s --subsamp_freq 20
    python bc.py Walker2d-v1    25 --test_rollouts 50 --eval_freq 100 --train_iters 20001 --seed $s --subsamp_freq 20
done
for e in 80 160 240; do
    for s in 0 1 2; do
        python bc.py Humanoid-v1 $e --test_rollouts 50 --eval_freq 100 --train_iters 20001 --seed $s --subsamp_freq 20
    done
done
