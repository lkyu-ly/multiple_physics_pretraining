#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH -J demo
#SBATCH --open-mode=append
#SBATCH -C h100


export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=6

master_node=$SLURMD_NODENAME
VENVDIR=/venvs/
run_name="demo"
config="basic_config"   # options are "basic_config" for all or swe_only/comp_only/incomp_only/swe_and_incomp
yaml_config="./config/mpp_avit_b_config.yaml"


module purge
module load modules/2.2-20230808
module load gcc openmpi/4.0.7 python-mpi/3.11.2

module load hdf5/mpi-1.12.2
module load cuda/12.1 cudnn/cuda12-8.9.0 nccl/cuda12.1-2.18.1 

# conda activate pdebench
source $VENVDIR/pdebench_venv/bin/activate
set -x

GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

python -m paddle.distributed.launch \
	--gpus=$GPUS \
	train_basic.py --run_name $run_name --config $config --yaml_config $yaml_config --use_ddp
