#!/bin/bash
# --- ログインノード ---
#cd ~/data
#qsub -I -P gcg51524 -q rt_HF -l select=1 -l walltime=00:30:00 <<'EOS'
# --- 計算ノードに移行後の処理 ---
#cd ~/data ここまで自分で打つ
module load python/3.12/3.12.9 cuda/12.1/12.1.1 nccl/2.23/2.23.4-1
source delsa_venv/bin/activate
export WANDB_API_KEY=2a8cba4b07893aff4e518befdc45d23017d1f145
torchrun --standalone --nproc_per_node=8 train_ddp.py
EOS
