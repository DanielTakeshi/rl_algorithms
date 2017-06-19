#!/bin/bash
set -eux
for e in 4 11 18 25; do
    for s in 0 1 2; do
        python bc.py Ant-v1         $e --seed $s
        python bc.py HalfCheetah-v1 $e --seed $s
        python bc.py Hopper-v1      $e --seed $s
        python bc.py Walker2d-v1    $e --seed $s
    done
done
