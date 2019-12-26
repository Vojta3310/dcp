#!/bin/bash

module load Python/3.7.2-fosscuda-2019a
module load PyTorch/1.2.0-fosscuda-2019a-Python-3.7.2
python3 main.py --exp_name=odometry_test2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --batch_size 4 --test_batch_size 4 --num_points 30000
