"""
Compare different classifiers in their accuracy in predicting tangrams from utterances
"""

import numpy as np
import pandas as pd
import pickle
import os
import torch
from pyprojroot import here
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from get_stimulus_logits import main as get_stimulus_logits
from evaluate_pretrained_model import TANGRAM_NAMES
from xgboost import XGBClassifier
from util import DotDict

classifiers = {
    "RandomForest": {
        "class": RandomForestClassifier,
        "params": [
            {"n_estimators": 10},
            {"n_estimators": 50},
            {"n_estimators": 100},
            {"n_estimators": 500},
        ],
    },
    "LogisticRegression": {
        "class": LogisticRegression,
        "params": [
            {"penalty": None},
            {"penalty": "l2"},
        ],
    },
    "MLP": {
        "class": MLPClassifier,
        "params": [
            {"hidden_layer_sizes": (32,)},
            {"hidden_layer_sizes": (32, 32)},
            {"hidden_layer_sizes": (100,)},
            {"hidden_layer_sizes": (100, 100)},
            {"hidden_layer_sizes": (512,)},
            {"hidden_layer_sizes": (1028,)},
        ],
    },
    "xgb": {
        "class": XGBClassifier,
        "params": [
            {},
            {"n_estimators": 10},
            {"n_estimators": 100},
        ],
    },
}


def try_classifiers(features, labels, n_folds):
    rows = []
    for name, classifier in classifiers.items():
        print(f"Trying {name}")
        for params in classifier["params"]:
            print(f"params: {params}")
            clf = classifier["class"](**params)
            scores = cross_val_score(clf, features, labels, cv=n_folds)
            print(f"Accuracy: {np.mean(scores)}")
            rows.append(
                {
                    "classifier": name,
                    "params": params,
                    "accuracy": np.mean(scores),
                }
            )

    return pd.DataFrame(rows)


def main(args):
    # load the data
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
    labels = [TANGRAM_NAMES.index(l) for l in df_logits["label"].values]

    df = try_classifiers(feats, labels, n_folds=10)
    df.to_csv(
        here(f"data/classifier_comparison-{model_name_for_file}.csv"), index=False
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
