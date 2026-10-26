#!/bin/bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
export TF_ENABLE_ONEDNN_OPTS=0 KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd ~/hanabi/drl_for_hanabi/algorithms
D=/home/percy/hanabi/drl_for_hanabi/experiments/policy_params/v2
r=$1
shift
A="--mode population --run-name $r --partners bram_simple bram_best"
B="--train-external sad --no-selfplay --epochs 3000"
C="--init-from $D/s2_replay/policy_final.pth"
until python population_training_v3.py $A $B $C "$@"
do
  sleep 5
done
