#!/bin/bash
# usage: ./run_v5.sh RUN_NAME [extra flags]   e.g. ./run_v5.sh v5_ppo --ppo
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
export TF_ENABLE_ONEDNN_OPTS=0 KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd ~/hanabi/drl_for_hanabi/algorithms
r=$1
shift
INIT=/home/percy/hanabi/drl_for_hanabi/experiments/policy_params/v2/s3_replay/policy_final.pth
until python population_training_v5.py --mode population --run-name $r \
      --init-from $INIT --partners bram_simple bram_best --train-external obl sad \
      --no-selfplay --epochs 1000 --batch-size 4000 --lr 3e-4 "$@" >> $r.log 2>&1
do
  sleep 5
done
