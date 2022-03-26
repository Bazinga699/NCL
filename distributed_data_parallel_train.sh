#!/usr/bin/env bash
export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo


PYTHON=${PYTHON:-"python"}
CONFIG=$1
NUM_GPUS=$2
GPUS=$3
PORT=${PORT:-29657}


CUDA_VISIBLE_DEVICES=$GPUS $PYTHON -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=$PORT \
        main/multi_network_train_final.py --cfg $CONFIG ${@:4}
