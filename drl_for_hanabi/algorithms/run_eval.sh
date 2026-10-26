#!/bin/bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
export TF_ENABLE_ONEDNN_OPTS=0 KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
r=$1
P="--partners bram_simple bram_best"
X="--no-selfplay --external obl sad --eval-episodes 40"
until python population_training_v3.py --mode eval --run-name $r $P $X > eval_$r.log 2>&1 && grep -q bombed eval_$r.log
do
  sleep 5
done
