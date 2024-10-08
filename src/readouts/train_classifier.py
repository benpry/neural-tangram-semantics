"""
Train a linear layer to predict the tangram from the CLIP similarity scores
"""
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from src.util import DotDict
import numpy as np
import pandas as pd
from src.evaluate_pretrained_model import TANGRAM_NAMES

def main(args):

    with open(args.data_filepath, "rb") as f:
        data = pickle.load(f)

    X = data["similarities"]
    y = data["labels"]

    all_preds = np.zeros(len(y))
    kf = KFold(n_splits=args.n_folds, shuffle=True)
    accuracies = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
        accuracies.append(clf.score(X_test, y_test))

        preds = clf.predict(X_test)
        all_preds[test_index] = [TANGRAM_NAMES.index(p) for p in preds]


    print(f"mean accuracy: {sum(accuracies) / len(accuracies)}")
    print(f"all accuracies: {accuracies}")
    all_preds = [TANGRAM_NAMES[int(p)] for p in all_preds]

    pd.DataFrame({
        "gameId": data["gameId"],
        "trialNum": data["trialNum"],
        "repNum": data["repNum"],
        "playerId": data["playerId"],
        "tangram": data["labels"],
        "predicted": all_preds
    }).to_csv("data/random_forest_stimulus_predictions.csv", index=False)

if __name__ == "__main__":

    args = DotDict(
        model_name = "openai/clip-vit-large-patch14",
        data_filepath = "data/similarities/similarities-openai_clip-vit-large-patch14.p",
        use_kilogram = False,
        n_folds = 10,
    )

    main(args)
