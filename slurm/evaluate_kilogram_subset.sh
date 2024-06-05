#!/bin/zsh

#SBATCH --job-name=run_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --gres gpu:1
#SBATCH --output=slurm-output/evaluation.log
#SBATCH --error=slurm-output/evaluation.err
#SBATCH --nodelist=cocoflops1

source ~/.zshrc
cd ~/neural-tangram-semantics
conda activate neural-tangrams
python src/evaluate_pretrained_model.py --use_kilogram --model_name /scr/benpry/kilogram-models/clip_controlled/parts+color/model1.pth
python src/evaluate_pretrained_model.py --use_kilogram --model_name /scr/benpry/kilogram-models/clip_controlled/augmented/model2.pth
python src/evaluate_pretrained_model.py --use_kilogram --model_name /scr/benpry/kilogram-models/clip_controlled/whole+black/model0.pth
python src/evaluate_pretrained_model.py --use_kilogram --model_name /scr/benpry/kilogram-models/clip_random/parts+color/model1.pth
python src/evaluate_pretrained_model.py --use_kilogram --model_name /scr/benpry/kilogram-models/clip_random/augmented/model2.pth
python src/evaluate_pretrained_model.py --use_kilogram --model_name /scr/benpry/kilogram-models/clip_random/whole+black/model0.pth