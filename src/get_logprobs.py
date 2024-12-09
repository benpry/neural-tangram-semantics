"""
This file gets surprisals from a conditional generation VLM
"""

import pandas as pd
import torch
from vllm import LLM, SamplingParams
from util import DotDict
from pyprojroot import here
from PIL import Image


def get_prompts(data_filepath, image_type):
    df = pd.read_csv(here(data_filepath))

    if image_type is None:
        image = None
    elif image_type == "allshapes":
        image = Image.open(here("data/tangrams/all-tangrams-screenshot.png"))
    else:
        raise ValueError(f"Invalid image type: {image_type}")

    return list(df["words"]), image


def main(config):

    sampling_params = SamplingParams(temperature=1.0, prompt_logprobs=0)
    model = LLM(
        config.model_name,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.8,
        max_model_len=1024,
        max_num_seqs=16,
        enforce_eager=True,
        # tensor_parallel_size=4,
    )

    prompts, image = get_prompts(config.data_filepath, config.image_type)

    if config.image_type is None:
        outputs = model.generate(
            prompts,
            sampling_params,
        )
    else:
        outputs = model.generate(
            [
                {"prompt": prompt, "multi_modal_data": {"image": image}}
                for prompt in prompts
            ],
            sampling_params,
        )

    rows = []
    for prompt_idx in range(len(prompts)):
        output = outputs[prompt_idx]
        word_level_logprobs = []
        curr_word = ""
        curr_logprob = 0
        for token in output.prompt_logprobs[1:]:
            assert len(token.values()) == 1
            token = list(token.values())[0]
            if not token.decoded_token or token.decoded_token[0] != " ":
                curr_word += token.decoded_token
                curr_logprob += token.logprob
            else:
                word_level_logprobs.append((curr_word, curr_logprob))
                curr_word = token.decoded_token[1:]
                curr_logprob = token.logprob

        if curr_word:
            word_level_logprobs.append((curr_word, curr_logprob))

        for i, (word, logprob) in enumerate(word_level_logprobs):
            rows.append(
                {
                    "full_text": prompts[prompt_idx],
                    "word_idx": i,
                    "word": word,
                    "logprob": logprob,
                }
            )

    df_logprobs = pd.DataFrame(rows)

    input_filename = config.data_filepath.split("/")[-1].split(".")[0]
    if config.image_type is None:
        df_logprobs.to_csv(
            here(
                f"data/logprobs/{input_filename}-logprobs-{config.model_name.replace('/', '--')}.csv"
            ),
            index=False,
        )
    else:
        df_logprobs.to_csv(
            here(
                f"data/logprobs/{input_filename}-logprobs-{config.model_name.replace('/', '--')}-{config.image_type}.csv"
            ),
            index=False,
        )


if __name__ == "__main__":

    text_config = DotDict(
        {
            "model_name": "meta-llama/Llama-3.1-8B",
            "data_filepath": "data/raw-data/expt_4_games_concat.csv",
            "image_type": None,
        }
    )

    # main(text_config)

    multimodal_config = DotDict(
        {
            "model_name": "meta-llama/Llama-3.2-11B-Vision",
            "data_filepath": "data/raw-data/expt_4_games_concat.csv",
            "image_type": "allshapes",
        }
    )

    main(multimodal_config)

    # TODO: run on the role data
