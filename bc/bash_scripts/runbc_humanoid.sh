#!/bin/bash
set -eux
for e in 80 160 240; do
    for s in 0 1 2; do
        python bc.py Humanoid-v1 $e --seed $s
    done
done
