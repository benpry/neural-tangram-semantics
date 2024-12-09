"""
Run a classifier on each of the stimuli
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


classifier_name = "MLP_100_100"


def get_probs_from_classifier_kfold(feats, labels, n_folds=10):
    clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=2000)
    kf = KFold(n_splits=n_folds, shuffle=False)
    rows = []
    for train_idx, test_idx in kf.split(feats):
        train_feats, test_feats = feats[train_idx], feats[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]
        clf.fit(train_feats, train_labels)
        clf
        test_probs = clf.predict_proba(test_feats)
        for i, prob_vec in enumerate(test_probs):
            row = {f"p_{TANGRAM_NAMES[j]}": p for j, p in enumerate(prob_vec)}
            row["raw_logits"] = test_feats[i]
            row["tangram"] = TANGRAM_NAMES[test_labels[i]]
            rows.append(row)

    return pd.DataFrame(rows)


def get_probs_from_classifier(feats, labels):
    clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=2000)
    clf.fit(feats, labels)
    probs = clf.predict_proba(feats)
    rows = []
    for i, prob_vec in enumerate(probs):
        row = {f"p_{TANGRAM_NAMES[j]}": p for j, p in enumerate(prob_vec)}
        row["raw_logits"] = feats[i]
        row["tangram"] = TANGRAM_NAMES[labels[i]]
        rows.append(row)

    return pd.DataFrame(rows)


def main(args):

    model_name_for_file = args.model_name.replace("/", "--")
    data_filepath = here(f"data/stimulus-logits/logits-{model_name_for_file}.csv")
    if not os.path.exists(data_filepath):
        get_stimulus_logits(args)
    df_logits = pd.read_csv(data_filepath)

    feats = np.array(
        [
            np.fromstring(x.strip().replace("\n", "")[1:-1], sep=" ")
            for x in df_logits["logits"].values
        ]
    )
    labels = np.array([TANGRAM_NAMES.index(l) for l in df_logits["label"].values])

    if args.kfold:
        df_preds = get_probs_from_classifier_kfold(feats, labels, n_folds=10)
    else:
        df_preds = get_probs_from_classifier(feats, labels)

    df_preds["utterance"] = df_logits["utterance"]
    df_preds["gameId"] = df_logits["gameId"]
    df_preds["trialNum"] = df_logits["trialNum"]
    df_preds["repNum"] = df_logits["repNum"]
    df_preds["playerId"] = df_logits["playerId"]
    if args.kfold:
        df_preds.to_csv(
            here(
                f"data/stimulus_predictions/{classifier_name}-{model_name_for_file}-probs-kfold.csv"
            ),
            index=False,
        )
    else:
        df_preds.to_csv(
            here(
                f"data/stimulus_predictions/{classifier_name}-{model_name_for_file}-probs.csv"
            ),
            index=False,
        )


if __name__ == "__main__":
    args = DotDict(
        {
            "model_name": "openai/clip-vit-large-patch14",
            "data_filepath": "speaker_utterances.csv",
            "batch_size": 32,
            "use_kilogram": False,
            "kfold": False,
        }
    )
    main(args)
