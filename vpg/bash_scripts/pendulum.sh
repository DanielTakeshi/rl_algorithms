#!/bin/bash
python main.py Pendulum-v0 --vf_type linear --seed 4
python main.py Pendulum-v0 --vf_type nn --seed 4
python main.py Pendulum-v0 --vf_type linear --seed 6
python main.py Pendulum-v0 --vf_type nn --seed 6
python main.py Pendulum-v0 --vf_type linear --seed 8
python main.py Pendulum-v0 --vf_type nn --seed 8
