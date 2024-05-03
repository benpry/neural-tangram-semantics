"""
Evaluate all the pretrained kilogram models
"""
import os
import glob
from evaluate_pretrained_model import main as evaluate_model
from pyprojroot import here
import pandas as pd
from util import DotDict


eval_args = DotDict(
    use_kilogram=True,
    n_batches="all",
    batch_size=100,
    data_filepath="speaker_utterances.csv",
)


def main():
    models_dir = os.environ["SCR_ROOT_DIR"] + "/kilogram-models"
    rows = []
    # traverse the directory to find the pretrained model paths
    for model_path in glob.glob(models_dir + "/clip_*/*/model*.pth"):
        print(f"evaluating model: {model_path}")
        eval_args.model_name = model_path
        evaluation = evaluate_model(eval_args)
        rows.append({**evaluation, "model": model_path})

    df = pd.DataFrame(rows)
    df.to_csv("evaluation_results.csv")


if __name__ == "__main__":
    main()
