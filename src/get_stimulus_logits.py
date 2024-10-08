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

    model, processor = set_up_model(args.model_name)
    tangrams = load_tangrams(12)
    tangrams_list = [tangrams[t] for t in TANGRAM_NAMES]

    df_data = pd.read_csv(here(f"data/{args.data_filepath}"))
    batches = get_batches(
        df_data, processor, args.use_kilogram, batch_size=args.batch_size
    )
    rows = []
    for batch in tqdm(batches):
        print(f"batch: {batch}")
        _, logits = get_model_probs(
            model, processor, batch, tangrams_list, use_kilogram=args.use_kilogram
        )
        rows.extend(
            {
                "utterance": batch["utterance"],
                "label": batch["label"],
                "logits": logits,
            }
        )

    model_name_file = args.model_name.replace("/", "--")
    pd.DataFrame(rows).to_csv(here(f"data/logits-{model_name_file}.csv"))
