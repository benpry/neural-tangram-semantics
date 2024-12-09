#!/bin/zsh

#SBATCH --job-name=compute_classifier_probs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --gres gpu:1
#SBATCH --output=slurm-output/classifier-probs.out
#SBATCH --error=slurm-output/classifier-probs.err
#SBATCH --nodelist=cocoflops-hgx-1

source ~/.zshrc
cd ~/neural-tangram-semantics
conda activate neural-tangrams
python src/stimulus_probabilities_from_classifier.py