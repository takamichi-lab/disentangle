#!/bin/sh
# run_abci_rtHF_ddp.sh — ABCI 3.0 single-node 8GPU job (rt_HF)
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -P <your_group>

cd ${PBS_O_WORKDIR}

# Environment (adjust modules to your ABCI account)
source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1

# Optional: activate your venv
source ~/venvs/delsa/bin/activate

export OMP_NUM_THREADS=8
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

# Launch DDP (per-GPU batch is set in config.yaml)
torchrun --standalone --nproc_per_node=8 /mnt/data/train_ddp.py /mnt/data/config.yaml
