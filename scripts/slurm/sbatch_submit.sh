#!/bin/bash
# Submit an ifarm job to a GPU。
# Usage: sh sbatch_submit.sh

# todo: add some cli input params
IFARM_SCRIPT_DIR=/home/xmei/projects/pytorch-paradnn/scripts/slurm

sbatch --gres gpu:T4:1 ${IFARM_SCRIPT_DIR}/submit_ifarm_gpu.slurm  # change gres name
