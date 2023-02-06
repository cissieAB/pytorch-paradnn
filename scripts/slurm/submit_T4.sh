#!/bin/bash
# Submit an ifarm job to T4 GPU。
# Usage: sh submit_T4.sh

# todo: add some cli input params
SCRIPT_DIR=/home/xmei/projects/pytorch-paradnn/scripts/slurm

sbatch --gres gpu:T4:1 ${SCRIPT_DIR}/submit_ifarm_gpu.slurm
