#!/bin/bash
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_INSTALL_PATH="/usr/local/cuda-12.8"
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
export MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi/lib # 替换为你的 MPI 安装路径
export LD_LIBRARY_PATH=$MPI_DIR:$LD_LIBRARY_PATH

deepspeed --num_nodes=1 --num_gpus=1 src/train_bge_base_with_emotion.py --deepspeed --deepspeed_config ds_configs/ds_config_with_emotion.json