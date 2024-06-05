"""
Evaluate a pretrained CLIP model on the tangram data
"""

from argparse import ArgumentParser
from string import ascii_uppercase
from pyprojroot import here
from PIL import Image
import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from kilogram_clip import FTCLIP, CLIPPreprocessor
from tqdm import tqdm

TANGRAM_NAMES = ascii_uppercase[:12]


def load_tangrams(n):
    tangrams = {}
    tangram_names = ascii_uppercase[:n]
    for tangram_letter in tangram_names:
        tangram = Image.open(here(f"data/tangrams/tangram_{tangram_letter}.png"))
        tangrams[tangram_letter] = tangram

    return tangrams


def set_up_model(model_name, use_kilogram=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_kilogram:
        checkpoint = torch.load(model_name, map_location=device)
        model = FTCLIP()
        model.load_state_dict(checkpoint["model_state_dict"])
        processor = CLIPPreprocessor(device=device)
    else:
        model = CLIPModel.from_pretrained(model_name, device_map=device)
        processor = CLIPProcessor.from_pretrained(model_name, device_map=device)
        print(f"model device: {model.device}")

    return model, processor


def get_batches(df, processor, use_kilogram, batch_size=10, n_batches="all"):
    """
    Get a batch of utterances, one for each tangram
    """

    # filter out utterances that are too long for CLIP
    if use_kilogram:
        df["too_long"] = df["text"].apply(
            lambda x: not torch.is_tensor(processor.preprocess_texts([x]))
        )
    else:
        df["too_long"] = df["text"].apply(lambda x: len(processor(x).input_ids) > 77)
    df_filtered = df[~df["too_long"]]

    # get the batches ready
    if isinstance(n_batches, int):
        df_filtered = df_filtered.sample(n_batches * batch_size)
    else:
        n_batches = len(df_filtered) // batch_size
    df_batches = df_filtered[
        ["gameId", "trialNum", "repNum", "playerId", "text", "tangram"]
    ]
    batches = []
    for i in range(n_batches):
        batch = df_batches.iloc[i * batch_size : (i + 1) * batch_size]
        batches.append(
            {
                "gameId": batch["gameId"].tolist(),
                "trialNum": batch["trialNum"].tolist(),
                "repNum": batch["repNum"].tolist(),
                "playerId": batch["playerId"].tolist(),
                "utterance": batch["text"].tolist(),
                "label": batch["tangram"].tolist(),
            }
        )

    return batches


def get_model_probs(model, processor, batch, tangrams_list, use_kilogram=False):
    """
    Get the model's predictions for each tangram
    """
    # compile the inputs
    utterances = batch["utterance"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_kilogram:
        with torch.no_grad():
            processed_images = processor.preprocess_images(tangrams_list)
            text_encodings = processor.preprocess_texts(utterances)
            similarities = model(processed_images, text_encodings)
            probs = similarities.t().softmax(dim=1).detach().cpu().numpy()
    else:
        inputs = processor(
            text=utterances, images=tangrams_list, return_tensors="pt", padding=True
        )
        inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
        probs = logits_per_image.t().softmax(dim=1).detach().cpu().numpy()

    return probs


def get_accuracy_metrics(probs, labels):
    """
    Get the accuracy metrics for the model
    """
    label_idxs = [ord(label) - ord("A") for label in labels]
    correct_answer_probs = probs[np.arange(probs.shape[0]), label_idxs]
    mean_prob = np.mean(correct_answer_probs)
    argmax_probs = np.argmax(probs, axis=1)
    accuracy = np.mean(argmax_probs == label_idxs)

    return mean_prob, accuracy


def get_rows(batch, probs):
    """
    convert a batch and probability matrix into rows of a dataframe
    """
    prob_dict = {f"p_{TANGRAM_NAMES[i]}": probs[:, i] for i in range(probs.shape[1])}
    model_predictions = [TANGRAM_NAMES[i] for i in np.argmax(probs, axis=1)]
    merged_dict = {**batch, **prob_dict, "prediction": model_predictions}

    return [dict(zip(merged_dict, t)) for t in zip(*merged_dict.values())]


def mean_ci_boot(data, statfunc=np.mean, n_samples=10000, ci=0.95):
    """
    Compute the mean and confidence interval of a statistic using bootstrapping
    """
    samples = np.random.choice(data, (n_samples, len(data)), replace=True)
    stats = statfunc(samples, axis=1)
    mean = np.mean(stats)
    lower = np.percentile(stats, (1 - ci) / 2 * 100)
    upper = np.percentile(stats, (1 + ci) / 2 * 100)

    return mean, lower, upper


def main(args):
    model, processor = set_up_model(args.model_name, use_kilogram=args.use_kilogram)
    tangrams = load_tangrams(12)
    tangrams_list = [tangrams[t] for t in TANGRAM_NAMES]

    df_data = pd.read_csv(here(f"data/{args.data_filepath}"))
    mean_probs = []
    accuracies = []
    batches = get_batches(
        df_data, processor, args.use_kilogram, batch_size=args.batch_size
    )
    rows = []
    for batch in tqdm(batches):
        probs = get_model_probs(
            model, processor, batch, tangrams_list, use_kilogram=args.use_kilogram
        )
        rows.extend(get_rows(batch, probs))
        # print(f"probs: {probs}")
        mean_p, acc = get_accuracy_metrics(probs, batch["label"])
        accuracies.append(acc)
        mean_probs.append(mean_p)

    mean_mean_prob = np.mean(mean_probs)
    _, mean_prob_lower, mean_prob_upper = mean_ci_boot(mean_probs)
    mean_accuracy = np.mean(accuracies)
    _, accuracy_lower, accuracy_upper = mean_ci_boot(accuracies)

    model_name_file = args.model_name.replace("/", "|")
    pd.DataFrame(rows).to_csv(
        here(
            f"data/stimulus_predictions/clip_stimulus_level_predictions_model-{model_name_file}.csv"
        ),
        index=False,
    )

    return {
        "mean_mean_prob": mean_mean_prob,
        "mean_prob_lower": mean_prob_lower,
        "mean_prob_upper": mean_prob_upper,
        "mean_accuracy": mean_accuracy,
        "accuracy_lower": accuracy_lower,
        "accuracy_upper": accuracy_upper,
    }


parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
parser.add_argument("--data_filepath", type=str, default="speaker_utterances.csv")
parser.add_argument("--batch_size", default=100)
parser.add_argument("--n_batches", default="all")
parser.add_argument("--use_kilogram", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    result = main(args)
    print(result)
