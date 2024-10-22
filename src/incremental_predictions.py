"""
Get stimulus-level predictions for incremental versions of the utterances
"""

import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from evaluate_pretrained_model import TANGRAM_NAMES
from pyprojroot import here
from get_stimulus_logits import main as get_stimulus_logits
from util import DotDict


def get_incremental_probs_from_classifier(df, df_incremental, n_folds=10):
    indices = df.index.values
    clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=2000)
    kf = KFold(n_splits=n_folds, shuffle=False)
    rows = []
    for train_idx, test_idx in kf.split(indices):
        df_train, df_test = df.loc[train_idx], df.loc[test_idx]

        # Train the classifier on the train set
        train_feats = np.array(
            [
                np.fromstring(x.strip().replace("\n", "")[1:-1], sep=" ")
                for x in df_train["logits"].values
            ]
        )
        train_labels = np.array(
            [TANGRAM_NAMES.index(l) for l in df_train["label"].values]
        )

        # Merge in the incremental utterances
        clf.fit(train_feats, train_labels)

        df_test = df_test[["gameId", "trialNum", "repNum"]]
        df_test_incremental = pd.merge(
            df_test, df_incremental, on=("gameId", "trialNum", "repNum"), how="inner"
        )
        print(f"test incremental shape: {df_test_incremental.shape}")

        # incremental_test_feats = np.array(df_test_incremental["logits"].values)
        print(f"dtype: {df_test_incremental["logits"].dtype}")
        print(df_test_incremental.iloc[0]["logits"])
        print(type(df_test_incremental.iloc[0]["logits"]))
        incremental_test_feats = np.array(
            [
                np.fromstring(x.strip().replace("\n", "")[1:-1], sep=" ")
                for x in df_test_incremental["logits"].values
            ]
        )
        print(f"test feats shape: {incremental_test_feats.shape}")

        test_probs = clf.predict_proba(incremental_test_feats)
        for i, prob_vec in enumerate(test_probs):
            row = {f"p_{TANGRAM_NAMES[j]}": p for j, p in enumerate(prob_vec)}
            row["raw_logits"] = incremental_test_feats[i]
            row["tangram"] = df_test_incremental.iloc[i]["label"]
            row["gameId"] = df_test_incremental.iloc[i]["gameId"]
            row["trialNum"] = df_test_incremental.iloc[i]["trialNum"]
            row["repNum"] = df_test_incremental.iloc[i]["repNum"]
            row["utterance"] = df_test_incremental.iloc[i]["utterance"]
            rows.append(row)

    return pd.DataFrame(rows)


def main(args):

    model_name_for_file = args.model_name.replace("/", "--")
    data_filepath = here(f"data/stimulus-logits/logits-{model_name_for_file}.csv")
    if not os.path.exists(data_filepath):
        get_stimulus_logits(args)
    df_logits = pd.read_csv(data_filepath)
    incremental_filepath = here(
        f"data/stimulus-logits/incremental-logits-{model_name_for_file}.csv"
    )
    if not os.path.exists(incremental_filepath):
        incremental_args = args.copy()
        incremental_args.data_filepath = "incremental_utterances.csv"
        get_stimulus_logits(incremental_args)
    df_incremental = pd.read_csv(incremental_filepath)

    df_preds = get_incremental_probs_from_classifier(
        df_logits, df_incremental, n_folds=10
    )
    df_preds.to_csv(
        here(f"data/stimulus_predictions/incremental-probs.csv"),
        index=False,
    )


if __name__ == "__main__":
    args = DotDict(
        {
            "model_name": "openai/clip-vit-large-patch14",
            "data_filepath": "speaker_utterances.csv",
            "batch_size": 32,
            "use_kilogram": False,
        }
    )
    main(args)
