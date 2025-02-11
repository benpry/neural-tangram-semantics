#!/bin/zsh

#SBATCH --job-name=run_classifier_comparison
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --output=slurm-output/classifier-comparison.log
#SBATCH --error=slurm-output/classifier-comparison.err
#SBATCH --nodelist=cocoflops1

source ~/.zshrc
cd ~/neural-tangram-semantics
conda activate neural-tangrams
python src/classifier_comparison.py