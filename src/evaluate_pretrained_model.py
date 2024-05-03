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
import scipy
from kilogram_clip import FTCLIP, CLIPPreprocessor


def load_tangrams(n):
    tangrams = {}
    tangram_names = ascii_uppercase[:n]
    for tangram_letter in tangram_names:
        tangram = Image.open(here(f"data/tangrams/tangram_{tangram_letter}.png"))
        tangrams[tangram_letter] = tangram

    return tangrams


def set_up_model(model_name, use_kilogram=False):
    if use_kilogram:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(here(f"models/model0.pth"), map_location=device)
        model = FTCLIP()
        model.load_state_dict(checkpoint["model_state_dict"])
        processor = CLIPPreprocessor(device=device)
    else:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

    return model, processor


def get_batches(df, batch_size=10, n_batches="all"):
    """
    Get a batch of utterances, one for each tangram
    """

    # filter out utterances that are too long for CLIP
    df["too_long"] = df["text"].apply(lambda x: len(x) > 77)
    df_filtered = df[~df["too_long"]]

    # get the batches ready
    if isinstance(n_batches, int):
        df_filtered = df_filtered.sample(n_batches * batch_size)
    else:
        n_batches = len(df_filtered) // batch_size
    df_batches = df_filtered[["text", "tangram"]]
    batches = []
    for i in range(n_batches):
        batch = df_batches.iloc[i * batch_size : (i + 1) * batch_size]
        batches.append(
            {
                "utterances": batch["text"].tolist(),
                "labels": batch["tangram"].tolist(),
            }
        )

    return batches


def get_model_probs(model, processor, batch, tangrams, use_kilogram=False):
    """
    Get the model's predictions for each tangram
    """
    # compile the inputs
    utterances = batch["utterances"]

    if use_kilogram:
        with torch.no_grad():
            processed_images = processor.preprocess_images(tangrams.values())
            text_encodings = processor.preprocess_texts(utterances)
            similarities = model(processed_images, text_encodings)
            probs = similarities.t().softmax(dim=1).detach().numpy()
    else:
        inputs = processor(
            text=utterances, images=tangrams, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).detach().numpy()

    return probs


def get_accuracy_metrics(probs, labels):
    """
    Get the accuracy metrics for the model
    """
    label_idxs = [ord(label) - ord("A") for label in labels]
    print(f"label_idxs: {label_idxs}")
    print(f"shape of probs: {probs.shape}")
    correct_answer_probs = probs[np.arange(probs.shape[0]), label_idxs]
    print(f"correct_answer_probs: {correct_answer_probs}")
    mean_prob = np.mean(correct_answer_probs)
    max_probs = np.argmax(probs, axis=1)
    accuracy = np.mean(max_probs == label_idxs)

    return mean_prob, accuracy


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

    df_data = pd.read_csv(here(f"data/{args.data_filepath}"))
    mean_probs = []
    accuracies = []
    batches = get_batches(df_data, n_batches=args.n_batches)
    for batch in batches:
        probs = get_model_probs(
            model, processor, batch, tangrams, use_kilogram=args.use_kilogram
        )
        print(f"probs: {probs}")
        mean_p, acc = get_accuracy_metrics(probs, batch["labels"])
        accuracies.append(acc)
        mean_probs.append(mean_p)

    mean_mean_prob = np.mean(mean_probs)
    _, mean_prob_lower, mean_prob_upper = mean_ci_boot(mean_probs)
    mean_accuracy = np.mean(accuracies)
    _, mean_prob_lower, mean_prob_upper = mean_ci_boot(accuracies)

    return {
        "mean_mean_prob": mean_mean_prob,
        "mean_prob_lower": mean_prob_lower,
        "mean_prob_upper": mean_prob_upper,
        "mean_accuracy": mean_accuracy,
        "accuracy_lower": mean_prob_lower,
        "accuracy_upper": mean_prob_upper,
    }


parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
parser.add_argument("--data_filepath", type=str, default="speaker_utterances.csv")
parser.add_argument("--n_batches", type=int, default=10)
parser.add_argument("--use_kilogram", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    result = main(args)
    print(result)
