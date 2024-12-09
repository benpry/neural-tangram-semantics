"""
Get logits for each tangram from a particular model
"""

import pandas as pd
from evaluate_pretrained_model import (
    set_up_model,
    get_batches,
    load_tangrams,
    get_model_probs,
    TANGRAM_NAMES,
)
from pyprojroot import here
from tqdm import tqdm


def main(args):

    # load the models and data
    model, processor = set_up_model(args.model_name)
    tangrams = load_tangrams(12)
    tangrams_list = [tangrams[t] for t in TANGRAM_NAMES]
    df_data = pd.read_csv(here(args.data_filepath))

    # if the data has a "partial" column, use that as the text
    if "text" not in df_data.columns and "partial" in df_data.columns:
        df_data["text"] = df_data["partial"].apply(str)
    batches = get_batches(
        df_data, processor, args.use_kilogram, batch_size=args.batch_size
    )
    rows = []
    for batch in tqdm(batches):
        _, logits = get_model_probs(
            model, processor, batch, tangrams_list, use_kilogram=args.use_kilogram
        )
        result = {
            "gameId": batch["gameId"],
            "trialNum": batch["trialNum"],
            "repNum": batch["repNum"],
            "utterance": batch["utterance"],
            "label": batch["label"],
            "logits": logits,
        }
        batch_list = [dict(zip(result, t)) for t in zip(*result.values())]
        rows.extend(batch_list)

    model_name_file = args.model_name.replace("/", "--")
    dataset_name = args.data_filepath.split("/")[-1].split(".")[0]
    output_filename = f"logits-{dataset_name}-{model_name_file}"
    pd.DataFrame(rows).to_csv(here(f"data/stimulus-logits/{output_filename}.csv"))
