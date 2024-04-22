"""
Evaluate a pretrained CLIP model on the tangram data
"""
from argparse import ArgumentParser
from string import ascii_uppercase
from pyprojroot import here
from PIL import Image
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import scipy

TANGRAM_NAMES = ascii_uppercase[:12]


def load_tangrams(n):
    tangrams = {}
    for tangram_letter in TANGRAM_NAMES:
        tangram = Image.open(here(f"data/tangrams/tangram_{tangram_letter}.png"))
        tangrams[tangram_letter] = tangram

    return tangrams


def set_up_model(model_name):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def get_eval_batch(df, n_tangrams=12):
    """
    Get a batch of utterances, one for each tangram
    """
    batch = {}
    for tangram_name in TANGRAM_NAMES:
        df_filtered = df[df["tangram"] == tangram_name]
        utterance = None
        # CLIP has a max text length of 77. We should find a way around this.
        while utterance is None or len(utterance) > 77:
            utterance = df_filtered.sample(1).iloc[0]["text"]
        batch[tangram_name] = utterance

    return batch


def get_model_probs(model, processor, batch, tangrams):
    """
    Get the model's predictions for each tangram
    """
    # compile the inputs
    labels = [batch[x] for x in TANGRAM_NAMES]
    tangrams = [tangrams[x] for x in TANGRAM_NAMES]

    inputs = processor(text=labels, images=tangrams, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).detach().numpy()

    return probs


def get_accuracy_metrics(probs):
    """
    Get the accuracy metrics for the model
    """
    mean_prob = np.trace(probs) / probs.shape[0]
    max_probs = np.argmax(probs, axis=1)
    accuracy = np.mean(max_probs == np.arange(probs.shape[0]))

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


parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
parser.add_argument("--data_filepath", type=str, default="speaker_utterances.csv")
parser.add_argument("--n_batches", type=int, default=100)

if __name__ == "__main__":
    args = parser.parse_args()
    model, processor = set_up_model(args.model_name)
    tangrams = load_tangrams(12)

    df_data = pd.read_csv(here(f"data/{args.data_filepath}"))
    mean_probs = []
    accuracies = []
    for i in range(args.n_batches):
        batch = get_eval_batch(df_data)
        mean_p, acc = get_accuracy_metrics(
            get_model_probs(model, processor, batch, tangrams)
        )
        accuracies.append(acc)
        mean_probs.append(mean_p)

    probs = get_model_probs(model, processor, batch, tangrams)

    mean_mean_prob = np.mean(mean_probs)
    _, lower, upper = mean_ci_boot(mean_probs)
    print(
        f"Mean probability of true utterance: {mean_mean_prob:.3f} [{lower:.3f}, {upper:.3f}] (chance: {1/12:.3f})"
    )
    mean_accuracy = np.mean(accuracies)
    _, lower, upper = mean_ci_boot(accuracies)
    print(
        f"Mean accuracy: {mean_accuracy:.3f} [{lower:.3f}, {upper:.3f}] (chance: {1/12:.3f})"
    )
