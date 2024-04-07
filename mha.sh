#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --partition=disc_dual_a100_students,gpu,gpu_a100
#SBATCH --cpus-per-task=64
#SBATCH --mem=80G
#SBATCH --output=outputs/hw6_%j_stdout.txt
#SBATCH --error=outputs/hw6_%j_stderr.txt
#SBATCH --time=06:00:00
#SBATCH --job-name=hw6
#SBATCH --mail-user=brandondmichaud@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504319/cs5043-hw6
#SBATCH --array=0

. /home/fagg/tf_setup.sh
conda activate tf
module load cuDNN/8.9.2.26-CUDA-12.2.0


python hw6_base.py -vv @exp.txt @mha.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK --dataset /home/fagg/datasets/pfam
