#!/bin/bash
set -eux
for e in 4 11 18 25; do
    for l in 0.05 0.01 0.005 0.001 0.0005; do
        for s in 0.05 0.01 0.005 0.001 0.0005; do
            python bc.py Hopper-v1 $e --batch_size 128 --lrate $l \
                --regu $s --train_iters 2000
        done
    done
done
