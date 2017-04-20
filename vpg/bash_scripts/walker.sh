#!/bin/bash
python main.py Walker2d-v1 --vf_type linear --seed 4 --n_iter 3000
python main.py Walker2d-v1 --vf_type nn     --seed 4 --n_iter 3000  
