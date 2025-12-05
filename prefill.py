import argparse
import csv

import torch
from vllm import (
    LLM,
    SamplingParams,
)
from vllm.inputs.data import token_inputs
from datasets import load_dataset

from utils import (
    load_tokenizer,
    get_tokenized_inputs,
)

DEFAULT_SEED = 0
DEFAULT_MAX_NEW_TOKENS = 512

MODEL_FAMILIES = [
    "llama_2",
    "qwen_3",
    "gemma_3",
]
GEN_KWARGS = {
    "llama_2": {
        "temperature": 0.9,
        "top_p": 0.6,
    },
    "qwen_3": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0,
    },
    "gemma_3": {
        "top_k": 64,
        "top_p": 0.95,
    },
}

# Argument parser for command line arguments
parser = argparse.ArgumentParser(
    description="Script for performing prefilling attacks."
)

# Model and dataset arguments
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="the HuggingFace Hub model ID or local path of the model to evaluate",
)
parser.add_argument(
    "--model_family",
    type=str,
    required=True,
    help="the model family",
    choices=MODEL_FAMILIES,
)
parser.add_argument(
    "--prefills_path",
    type=str,
    required=True,
    help="the path to the file containing the prefills",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=DEFAULT_MAX_NEW_TOKENS,
    help="the maximum number of new tokens to generate",
)
parser.add_argument(
    "--thinking",
    action="store_true",
    help="whether the model has a 'thinking' mode, which will be disabled",
)
parser.add_argument(
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    help="the random seed for reproducibility",
)
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="the path to save the generated outputs",
)

args = parser.parse_args()

torch.manual_seed(args.seed)

model = LLM(
    model=args.model,
)
tokenizer = load_tokenizer(
    model_name_or_path=args.model,
)

dataset = load_dataset(
    path="csv",
    data_files=args.prefills_path,
    split="train",
)

inputs_tokens = get_tokenized_inputs(
    prompts=dataset["prompt"],
    tokenizer=tokenizer,
    prefills=dataset["prefill"],
    padding=False,
    return_dict=False,
    return_tensors=None,
    **({"enable_thinking": False} if args.thinking else {})
)
inputs_no_prefill_tokens = get_tokenized_inputs(
    prompts=dataset["prompt"],
    tokenizer=tokenizer,
    padding=False,
    return_dict=False,
    return_tensors=None,
    **({"enable_thinking": False} if args.thinking else {})
)
inputs = [
    token_inputs(
        prompt_token_ids=tokens,
    ) for tokens in inputs_tokens
]

sampling_params = SamplingParams(
    **GEN_KWARGS[args.model_family],
    max_tokens=args.max_new_tokens,
    seed=args.seed,
)

outputs = model.generate(
    prompts=inputs,
    sampling_params=sampling_params,
    use_tqdm=True,
)

full_outputs = [
    tokenizer.decode(
        input_sequence[len(input_no_prefill_sequence):] + \
            output.outputs[0].token_ids,
        skip_special_tokens=True,
    ) for input_sequence, input_no_prefill_sequence, output in zip(
        inputs_tokens,
        inputs_no_prefill_tokens,
        outputs,
    )
]

with open(args.output_path, "w") as f:
    writer = csv.writer(f)

    writer.writerow(["prompt", "output"])

    for prompt, output in zip(
        dataset["prompt"],
        full_outputs,
    ):
        writer.writerow([prompt, output])