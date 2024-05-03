#!/bin/zsh

#SBATCH --job-name=kilogram_models_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --gres gpu:1
#SBATCH --output=slurm-output/kilogram_model_sweep.log
#SBATCH --error=slurm-output/kilogram_model_sweep.err
#SBATCH --nodelist=cocoflops1

source ~/.zshrc
cd ~/neural-tangram-semantics
conda activate neural-tangrams
python src/kilogram_model_sweep.py
