#!/bin/zsh

#SBATCH --job-name=compute_incremental_probs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --gres gpu:1
#SBATCH --output=slurm-output/incremental-probs.out
#SBATCH --error=slurm-output/incremental-probs.err
#SBATCH --nodelist=cocoflops1

source ~/.zshrc
cd ~/neural-tangram-semantics
conda activate neural-tangrams
python src/incremental_predictions.py