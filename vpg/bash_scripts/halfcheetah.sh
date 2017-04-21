#!/bin/bash
python main.py HalfCheetah-v1 --vf_type linear --seed 4 --n_iter 3000
python main.py HalfCheetah-v1 --vf_type nn     --seed 4 --n_iter 3000  
python main.py HalfCheetah-v1 --vf_type linear --seed 6 --n_iter 3000
python main.py HalfCheetah-v1 --vf_type nn     --seed 6 --n_iter 3000
python main.py HalfCheetah-v1 --vf_type linear --seed 8 --n_iter 3000
python main.py HalfCheetah-v1 --vf_type nn     --seed 8 --n_iter 3000
