"""
Generate a dataset of images and CLIP similarity scores
"""

import pickle

import numpy as np
import pandas as pd
import torch
from pyprojroot import here
from tqdm import tqdm

from src.evaluate_pretrained_model import (
    TANGRAM_NAMES,
    get_batches,
    load_tangrams,
    set_up_model,
)
from src.util import DotDict


def get_model_logits(model, processor, batch, tangrams_list, use_kilogram=False):
    # compile the inputs
    utterances = batch["utterance"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_kilogram:
        with torch.no_grad():
            processed_images = processor.preprocess_images(tangrams_list)
            text_encodings = processor.preprocess_texts(utterances)
            similarities = model(processed_images, text_encodings)
    else:
        inputs = processor(
            text=utterances, images=tangrams_list, return_tensors="pt", padding=True
        )
        inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            similarities = outputs.logits_per_image

    return similarities.t().detach().cpu().numpy()


def main(args):
    model, processor = set_up_model(args.model_name, use_kilogram=args.use_kilogram)
    tangrams = load_tangrams(12)
    tangrams_list = [tangrams[t] for t in TANGRAM_NAMES]
    df_data = pd.read_csv(here(f"data/{args.data_filepath}"))

    batches = get_batches(
        df_data, processor, args.use_kilogram, batch_size=args.batch_size
    )

    all_feats = []
    all_labels = []
    all_game_ids = []
    all_trial_nums = []
    all_rep_nums = []
    all_player_ids = []
    for batch in tqdm(batches):

        similarities = get_model_logits(
            model, processor, batch, tangrams_list, args.use_kilogram
        )
        all_feats.append(similarities)
        all_labels.extend(batch["label"])
        all_game_ids.extend(batch["gameId"])
        all_trial_nums.extend(batch["trialNum"])
        all_rep_nums.extend(batch["repNum"])
        all_player_ids.extend(batch["playerId"])

    all_feats = np.vstack(all_feats)
    all_labels = np.array(all_labels)


    print(f"shape of all_feats: {all_feats.shape}")
    print(f"length of all_labels: {len(all_labels)}")

    model_name_str = args["model_name"].replace("/", "_")
    with open(here(f"data/similarities/similarities-{model_name_str}.p"), "wb") as f:
        pickle.dump(
            {
                "similarities": all_feats,
                "labels": all_labels,
                "gameId": all_game_ids,
                "trialNum": all_trial_nums,
                "repNum": all_rep_nums,
                "playerId": all_player_ids,
            },
            f,
        )


if __name__ == "__main__":
    args = DotDict(
        model_name="openai/clip-vit-large-patch14",
        data_filepath="speaker_utterances.csv",
        use_kilogram=False,
        batch_size=100,
    )
    main(args)
