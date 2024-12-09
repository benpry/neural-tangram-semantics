#!/bin/zsh

#SBATCH --job-name=get_logprobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --gres gpu:1
#SBATCH --output=slurm-output/logprobs.log
#SBATCH --error=slurm-output/logprobs.err
#SBATCH --nodelist=cocoflops-hgx-1

source ~/.zshrc
cd ~/neural-tangram-semantics
conda activate vllm
python src/get_logprobs.py
