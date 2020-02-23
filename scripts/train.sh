#!/bin/bash
CUDA_VISIBLE_DEVICES=4 \
 ./train_network.py \
 /data/unagi0/kawana/workspace/ShapeNetCore.v2/02691156/ \
../train_output_plane/ \
--use_sq \
--lr 1e-4 \
--n_primitives 20 \
--train_with_bernoulli \
--dataset_type shapenet_v2 \
--use_chamfer \
--run_on_gpu  \
--use_deformations
