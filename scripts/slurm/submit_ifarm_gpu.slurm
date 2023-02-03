#!/bin/bash

#SBATCH -N 1  # number of nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000  # MB
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=pytorch-paradnn-log_%j.out
#SBATCH --job-name=pytorch-paradnn

set -euxo pipefail

srun nvidia-smi

env

CONDA_DIR=/apps/anaconda3/2021.05/etc/profile.d  # depend on the farm config
WK_DIR=/home/xmei/projects/pytorch-paradnn

source /etc/profile.d/modules.sh
module use /apps/modulefiles
module load anaconda3
which conda
conda --version

sh $CONDA_DIR/conda.sh
conda-env list
pwd
cd $WK_DIR
pwd
# A100 GPU requires higher CUDA version than the ifarm default
# This env is tested in Nov, 2022 and was working
source activate pytorch-cuda11_6  # "source activate" instead of "conda activate"

hostname_str=`hostname -s`
platform=${hostname_str}-gpu

# get input params

# warm up
srun python ${WK_DIR}/fc.py --use_gpu=1

data_type='f32'  # TODO: make this a param
outpath=./output/${platform}/fc_${SLURM_JOBID}
mkdir -p $outpath

use_gpu=1

# todo: develop features on AMP
# bash iteration on: use_gpu, input_type, layers, nodes, use_amp
srun python3 fc.py --use_gpu=${use_gpu} --input_type=${data_type}

# for layers in 4 8 16 32 64 128
for layers in 4
do
#  for nodes in 32 64 128 256 512 1024 2048 4096 8192
  for nodes in 32
  do
    name=benchmark_l_${layers}_n_${nodes}_${data_type}_${platform}
    srun python3 fc.py -b --layers=${layers} --nodes=${nodes} --use_gpu=${use_gpu} --input_type=${data_type}\
    >> ${outpath}/${name}.csv
  done
done
