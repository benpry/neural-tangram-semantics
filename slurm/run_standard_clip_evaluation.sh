#!/bin/zsh

#SBATCH --job-name=run_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --gres gpu:1
#SBATCH --output=slurm-output/standard-clip-evaluation.log
#SBATCH --error=slurm-output/standard-clip-evaluation.err
#SBATCH --nodelist=cocoflops1

source ~/.zshrc
cd ~/neural-tangram-semantics
conda activate neural-tangrams
python src/evaluate_pretrained_model.py --model_name openai/clip-vit-large-patch14
