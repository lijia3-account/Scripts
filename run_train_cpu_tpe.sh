#!/bin/bash
# ============================================================
# SLURM 脚本 — 天河 xyfree 分区（CPU 版）
# ============================================================
# 使用说明：
#   1. 把 DATA_PATH 改成你的数据文件实际路径
#   2. 把 ptpe_transformer_train.py 路径改成实际路径
#   3. chmod +x run_train_cpu.sh
#   4. sbatch run_train_cpu.sh
# ============================================================

#SBATCH --job-name=ptpe_transformer
#SBATCH --output=logs/ptpe_%j.out
#SBATCH --error=logs/ptpe_%j.err
#SBATCH --time=99:00:00        # 最多支持 99 小时，建议根据需要调整
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16    # 用 16 核（天河 xyfree 有 64 核，用一半）
#SBATCH --mem=64G             # 内存 64 GB（可用 512G 里的 64G）
## ⚠️ 不申请 GPU（全 CPU 运行，PyTorch 会自动用 CPU）⚠️

# ── 环境 ─────────────────────────────────────────────────────
# 根据你的集群实际 conda 路径修改
#module load anaconda3 2>/dev/null || true
#conda activate base 2>/dev/null || true

# ── ⚠️ 路径配置（改成你的实际路径）─────────────────────────────
# 示例：
# DATA_PATH="/home/szfy_whlxy_1/project/PTPE_control_all_data.txt"
# SCRIPT_DIR="/home/szfy_whlxy_1/Transformer/"
DATA_PATH="/HOME/szfy_whlxy/szfy_whlxy_1/AI/data/TPE_control_all_data.txt"     # ← 改这里
SCRIPT_DIR="/HOME/szfy_whlxy/szfy_whlxy_1/AI/Transformer"                 # ← 改这里

# ── 运行训练 ─────────────────────────────────────────────────
/HOME/szfy_whlxy/szfy_whlxy_1/miniconda3/envs/cputorch/bin/python ${SCRIPT_DIR}/ptpe_transformer_final_revised.py \
