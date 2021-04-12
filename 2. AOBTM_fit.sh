#!/bin/bash
set -e

K=10   # number of topics
W=3591   # vocabulary size

alpha=`echo "scale=3;50/$K"|bc`
beta=0.005
n_day=16

voca_pt=OBTM/input/voca.txt
dwid_dir=OBTM/input/doc_wids/
model_dir=OBTM/output/
method=obtm

## learning parameters p(z) and p(z|w)
echo "=============== Topic Learning ============="
make -C OBTM/src
n_iter=500
lam=0.8
OBTM/src/run obtm $K $W $alpha $beta $dwid_dir $n_day $model_dir $n_iter $lam


