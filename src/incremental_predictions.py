"""
Get stimulus-level predictions for incremental versions of the utterances
"""

import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from evaluate_pretrained_model import TANGRAM_NAMES
from pyprojroot import here
from get_stimulus_logits import main as get_stimulus_logits
from util import DotDict


def get_incremental_probs_from_classifier_kfold(df, df_incremental, n_folds=10):
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


def get_incremental_probs_from_classifier(df, df_incremental):
    clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=2000)
    rows = []

    # Train the classifier on the full utterances
    train_feats = np.array(
        [
            np.fromstring(x.strip().replace("\n", "")[1:-1], sep=" ")
            for x in df["logits"].values
        ]
    )
    train_labels = np.array([TANGRAM_NAMES.index(l) for l in df["label"].values])
    clf.fit(train_feats, train_labels)

    print(
        f"accuracy on non-incremental data: {accuracy_score(train_labels, clf.predict(train_feats))}"
    )

    # Run the classifier
    incremental_feats = np.array(
        [
            np.fromstring(x.strip().replace("\n", "")[1:-1], sep=" ")
            for x in df_incremental["logits"].values
        ]
    )
    test_probs = clf.predict_proba(incremental_feats)

    for i, prob_vec in enumerate(test_probs):
        row = {f"p_{TANGRAM_NAMES[j]}": p for j, p in enumerate(prob_vec)}
        row["raw_logits"] = incremental_feats[i]
        row["tangram"] = df_incremental.iloc[i]["label"]
        row["gameId"] = df_incremental.iloc[i]["gameId"]
        row["trialNum"] = df_incremental.iloc[i]["trialNum"]
        row["repNum"] = df_incremental.iloc[i]["repNum"]
        row["utterance"] = df_incremental.iloc[i]["utterance"]
        rows.append(row)

    return pd.DataFrame(rows)


def main(args):

    model_name_for_file = args.model_name.replace("/", "--")
    dataset_name = args.data_filepath.split("/")[-1].split(".")[0]
    # save the full utterances
    df_utterances = pd.read_csv(here(args.data_filepath))
    df_full = df_utterances.drop(
        columns=["partial", "partial_length"]
    ).drop_duplicates()
    df_full = df_full[df_full["text"].apply(lambda x: isinstance(x, str))]
    full_data_filepath = here(f"data/raw-data/{dataset_name}-full.csv")
    df_full.to_csv(full_data_filepath, index=False)

    full_logits_filepath = here(
        f"data/stimulus-logits/logits-{dataset_name}-full-{model_name_for_file}.csv"
    )
    # Compute logits for the full utterances
    if not os.path.exists(full_logits_filepath):
        full_args = args.copy()
        full_args.data_filepath = str(full_data_filepath)
        get_stimulus_logits(full_args)
    df_logits = pd.read_csv(full_logits_filepath)

    df_incremental = df_utterances.drop(columns=["text"]).rename(
        columns={"partial": "text"}
    )
    df_incremental = df_incremental[
        df_incremental["text"].apply(lambda x: isinstance(x, str))
    ]
    incremental_data_filepath = here(f"data/raw-data/{dataset_name}-incremental.csv")
    df_incremental.to_csv(incremental_data_filepath, index=False)

    # Compute logits for the incremental utterances
    incremental_logits_filepath = here(
        f"data/stimulus-logits/logits-{dataset_name}-incremental-{model_name_for_file}.csv"
    )
    if not os.path.exists(incremental_logits_filepath):
        incremental_args = args.copy()
        incremental_args.data_filepath = str(incremental_data_filepath)
        get_stimulus_logits(incremental_args)
    df_incremental = pd.read_csv(incremental_logits_filepath)

    # Compute the incremental predictions
    df_preds = get_incremental_probs_from_classifier(df_logits, df_incremental)
    df_preds.to_csv(
        here(f"data/stimulus_predictions/incremental-probs-{dataset_name}.csv"),
        index=False,
    )


if __name__ == "__main__":
    args = DotDict(
        {
            "model_name": "openai/clip-vit-large-patch14",
            "data_filepath": "data/raw-data/incremental_subset.csv",
            "batch_size": 32,
            "use_kilogram": False,
        }
    )
    main(args)
