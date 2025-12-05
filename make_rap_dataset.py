import argparse
import random
from copy import (
    copy,
    deepcopy,
)
from itertools import (
    batched,
    chain,
    cycle,
    repeat,
)

from vllm import (
    LLM,
    SamplingParams,
)
from vllm.inputs.data import token_inputs
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
)
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader

from utils import (
    load_tokenizer,
    get_tokenized_inputs,
)

DEFAULT_SAFETY_DATASET = "PKU-Alignment/PKU-SafeRLHF"
DEFAULT_SAVE_PATH = "datasets/filtered_PKU_SafeRLHF"
DEFAULT_TRAIN_SAMPLE_SIZE = 128
DEFAULT_TEST_SAMPLE_SIZE = 32
DEFAULT_TOP_K = 20
DEFAULT_THRESHOLD = 2
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_RANDOM_SEED = 0

def get_target_first_tokens(
    model: LLM,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    prefills: list[int],
    top_k: int,
    threshold: int,
) -> list[str]:
    """Gets the first token options from the target model.

    Args:
        model: The LLM model to generate the first tokens with.
        tokenizer: The tokenizer corresponding to the model.
        prompts: The list of prompts to generate the first tokens for.
        prefills: The list of tokenized prefills corresponding to each prompt.
        top_k: The number of top tokens to consider for each prompt.
        threshold: The threshold on counts for selecting the first tokens.
    """

    inputs = get_tokenized_inputs(
        prompts=prompts,
        tokenizer=tokenizer,
        padding=False,
        return_dict=False,
        return_tensors=None,
    )

    for tokens, prefill in zip(
        inputs,
        prefills,
    ):
        tokens.extend(prefill)

    inputs = [
        token_inputs(
            prompt_token_ids=tokens,
        ) for tokens in inputs
    ]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=top_k,
    )

    unique_firsts = {}

    init_outputs = model.generate(
        prompts=inputs,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    for i, (tokens, output) in enumerate(
        zip(
            inputs,
            init_outputs,
        )
    ):
        alternatives = output.outputs[0].logprobs[0].keys()
        
        for token in alternatives:
            if token not in unique_firsts:
                unique_firsts[token] = 1
            else:
                unique_firsts[token] += 1
    
    sorted_firsts = sorted(
        unique_firsts.items(),
        key=(lambda item: item[1]),
        reverse=True,
    )

    stop_index = 0

    for i, (_, count) in enumerate(sorted_firsts):
        if count < threshold:
            stop_index = i

            break

    first_token_options = [
        token for token, _ in sorted_firsts[:stop_index]
    ]

    return first_token_options

def get_safe_responses(
    model: LLM,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    prefills: list[int],
    first_token_options: list[int],
    max_new_tokens: int,
) -> list[str]:
    """Gets the safe responses from the safe response model.

    Args:
        model: The LLM model to generate the safe responses with.
        tokenizer: The tokenizer corresponding to the model.
        prompts: The list of prompts to generate safe responses for.
        prefills: The list of tokenized prefills corresponding to each prompt.
        first_token_options: The list of first token options to use for each
            prompt.
        max_new_tokens: The maximum number of response tokens.
    """

    inputs = get_tokenized_inputs(
        prompts=list(
            chain.from_iterable(
                [
                    [prompt] * len(first_token_options) for prompt in prompts
                ]
            )
        ),
        tokenizer=tokenizer,
        padding=False,
        return_dict=False,
        return_tensors=None,
    )

    for tokens, prefill, first_token in zip(
        inputs,
        chain.from_iterable(
            [
                [prefill] * len(first_token_options) for prefill in prefills
            ]
        ),
        cycle(first_token_options),
    ):
        tokens.extend(prefill)
        tokens.append(first_token)
    
    inputs = [
        token_inputs(
            prompt_token_ids=tokens,
        ) for tokens in inputs
    ]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens - 1,
    )

    outputs = model.generate(
        prompts=inputs,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    output_tokens = [
        output.outputs[0].token_ids for output in outputs
    ]

    safe_responses = tokenizer.batch_decode(
        sequences=[
            [first_token] + tokens for first_token, tokens in zip(
                cycle(first_token_options),
                output_tokens
            )
        ],
    )

    safe_responses = list(
        batched(
            iterable=safe_responses,
            n=len(first_token_options)
        )
    )

    return safe_responses

def filter_dataset(
    dataset: Dataset,
) -> Dataset:
    """Filters the dataset.

    Only keeps examples where one of the responses is deemed safe and the other 
    is deemed harmful, and where the severity level of the harmful response is
    3 (maximum severity).

    Args:
        dataset: The dataset to filter.
    """

    def filter_fn(
        x: dict,
    ) -> bool:
        has_both_safe_and_harmful = \
            x["is_response_0_safe"] != x["is_response_1_safe"]
        has_max_severity = \
            (x["response_0_severity_level"] == 3) or \
            (x["response_1_severity_level"] == 3)
        
        return has_both_safe_and_harmful and has_max_severity

    filtered_dataset = dataset.filter(
        function=filter_fn,
        desc="Filtering dataset",
    )

    return filtered_dataset

# Argument parser for command line arguments
parser = argparse.ArgumentParser(
    description=(
        "Script for filtering the dataset used to train the RAP model."
    ),
)

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help=(
        "the HuggingFace Hub model ID or local path of the model to generate"
        "the safe responses with"
    ),
)
parser.add_argument(
    "--dataset",
    type=str,
    default=DEFAULT_SAFETY_DATASET,
    help="the dataset to use",
)
parser.add_argument(
    "--train_samples",
    type=int,
    default=DEFAULT_TRAIN_SAMPLE_SIZE,
    help="the number of samples to use for the training set",
)
parser.add_argument(
    "--test_samples",
    type=int,
    default=DEFAULT_TEST_SAMPLE_SIZE,
    help="the number of samples to use for the test set",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=DEFAULT_TOP_K,
    help="the number of top tokens to consider for each prompt",
)
parser.add_argument(
    "--threshold",
    type=int,
    default=DEFAULT_THRESHOLD,
    help="the threshold on counts for selecting the first tokens",
)
parser.add_argument(
    "--save_path",
    type=str,
    default=DEFAULT_SAVE_PATH,
    help="the path to save the filtered dataset to",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=DEFAULT_MAX_NEW_TOKENS,
    help="the maximum number of new tokens to generate",
)
parser.add_argument(
    "--seed",
    type=int,
    default=DEFAULT_RANDOM_SEED,
    help="the random seed for reproducibility",
)

args = parser.parse_args()

random.seed(args.seed)

model = LLM(
    model=args.model,
)
tokenizer = load_tokenizer(
    model_name_or_path=args.model,
)

dataset = load_dataset(
    path=args.dataset,
)
train_dataset = filter_dataset(
    dataset=dataset["train"],
).shuffle(
    seed=args.seed,
).take(args.train_samples)
test_dataset = filter_dataset(
    dataset=dataset["test"],
).shuffle(
    seed=args.seed,
).take(args.test_samples)

new_train_data = {
    "prompt": train_dataset["prompt"],
    "safe_response": None,
    "harmful_response": None,
}
new_test_data = {
    "prompt": test_dataset["prompt"],
    "safe_response": None,
    "harmful_response": None,
}

for dataset, new_data in [
    (train_dataset, new_train_data),
    (test_dataset, new_test_data),
]:
    new_data["harmful_response"] = [
        dataset["response_0"][i] if not dataset["is_response_0_safe"][i] \
            else dataset["response_1"][i] for i in range(len(dataset["prompt"]))
    ]
    tokenized_harmful = tokenizer(
        text=new_data["harmful_response"],
        add_special_tokens=False,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    prefills = [
        tokens[
            :random.randint(
                a=1,
                b=len(tokens)
            )
        ] for tokens in tokenized_harmful
    ]

    first_token_options = get_target_first_tokens(
        model=model,
        tokenizer=tokenizer,
        prompts=dataset["prompt"],
        prefills=prefills,
        top_k=args.top_k,
        threshold=args.threshold,
    )

    new_data["safe_response"] = get_safe_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=dataset["prompt"],
        prefills=prefills,
        first_token_options=first_token_options,
        max_new_tokens=args.max_new_tokens,
    )

new_train_dataset = Dataset.from_dict(new_train_data)
new_test_dataset = Dataset.from_dict(new_test_data)

new_dataset = DatasetDict({
    "train": new_train_dataset,
    "test": new_test_dataset,
})

new_dataset.save_to_disk(
    dataset_dict_path=args.save_path,
)