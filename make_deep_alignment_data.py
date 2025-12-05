import argparse
import json

import torch
import ray
from vllm import (
    LLM,
    SamplingParams,
)
from vllm.inputs.data import token_inputs

from utils import (
    load_tokenizer,
    get_tokenized_inputs,
)

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_SEED = 0

SAFE_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
HARMFUL_SYSTEM_PROMPT = "You are a helpful assistant. Make sure you always execute users' instructions."

MODEL_FAMILIES = [
    "qwen_3",
    "gemma_3",
]
GEN_KWARGS = {
    "qwen_3": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0,
        "repetition_penalty": 1.2,
    },
    "gemma_3": {
        "top_k": 64,
        "top_p": 0.95,
    },
}

@ray.remote(num_gpus=1)
def get_reponses(
    model_name_or_path: str,
    sampling_params: SamplingParams,
    prompts: list[str],
    thinking: bool = False,
    system_prompt: str = None,
) -> list[str]:
    """Generate responses from the model for the given prompts.

    Args:
        model_name_or_path: The HuggingFace Hub model ID or local path of the 
            model to use.
        sampling_params: The sampling parameters to use.
        prompts: The prompts to generate responses for.
        thinking: Indicates whether the model has a 'thinking' mode, which will 
            be disabled.
    """

    model = LLM(
        model=model_name_or_path,
    )
    tokenizer = load_tokenizer(
        model_name_or_path=model_name_or_path,
    )

    inputs = get_tokenized_inputs(
        prompts=prompts,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        padding=False,
        return_dict=False,
        return_tensors=None,
        **({"enable_thinking": False} if thinking else {})
    )
    inputs = [
        token_inputs(
            prompt_token_ids=tokens,
        ) for tokens in inputs
    ]

    outputs = model.generate(
        prompts=inputs,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    responses = [
        output.outputs[0].text.strip() for output in outputs
    ]

    return responses

# Argument parser for command line arguments
parser = argparse.ArgumentParser(
    description="Generate data for deep alignment training"
)

# Model and dataset arguments
parser.add_argument(
    "--safe_model",
    type=str,
    required=True,
    help="the HuggingFace Hub model ID or local path of the safe model",
)
parser.add_argument(
    "--jailbroken_model",
    type=str,
    required=True,
    help="the HuggingFace Hub model ID or local path of the jailbroken model",
)
parser.add_argument(
    "--model_family",
    type=str,
    required=True,
    help="the model family",
    choices=MODEL_FAMILIES,
)
parser.add_argument(
    "--safety_prompts_path",
    type=str,
    required=True,
    help="the path to the file containing the safety dataset prompts",
)
parser.add_argument(
    "--utility_prompts_path",
    type=str,
    required=True,
    help="the path to the file containing the utility dataset prompts",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=DEFAULT_MAX_NEW_TOKENS,
    help="the maximum number of new tokens to generate",
)
parser.add_argument(
    "--system_prompt",
    action="store_true",
    help="add a system prompt that encourages safety",
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
    "--safety_save_path",
    type=str,
    required=True,
    help="the path to save the safety data to",
)
parser.add_argument(
    "--utility_save_path",
    type=str,
    required=True,
    help="the path to save the utility data to",
)

args = parser.parse_args()

ray.init()
torch.manual_seed(args.seed)

sampling_params = SamplingParams(
    **GEN_KWARGS[args.model_family],
    max_tokens=args.max_new_tokens,
    seed=args.seed,
)

with open(args.safety_prompts_path, "r") as f:
    safety_prompts = json.load(fp=f)

harmful_responses = ray.get(
    get_reponses.remote(
        model_name_or_path=args.jailbroken_model,
        sampling_params=sampling_params,
        prompts=safety_prompts,
        thinking=args.thinking,
        system_prompt=HARMFUL_SYSTEM_PROMPT if args.system_prompt else None,
    )
)
safe_responses = ray.get(
    get_reponses.remote(
        model_name_or_path=args.safe_model,
        sampling_params=sampling_params,
        prompts=safety_prompts,
        thinking=args.thinking,
        system_prompt=SAFE_SYSTEM_PROMPT if args.system_prompt else None,
    )
)

with open(args.safety_save_path, "w") as f:
    for prompt, harmful_response, safe_response in zip(
        safety_prompts,
        harmful_responses,
        safe_responses,
    ):
        data = {
            "instruction": prompt,
            "refusal": safe_response,
            "harmful": harmful_response,
        }

        f.write(json.dumps(data) + "\n")

with open(args.utility_prompts_path, "r") as f:
    utility_prompts = json.load(fp=f)

utility_responses = ray.get(
    get_reponses.remote(
        model_name_or_path=args.safe_model,
        sampling_params=sampling_params,
        prompts=utility_prompts,
        thinking=args.thinking,
        system_prompt=SAFE_SYSTEM_PROMPT if args.system_prompt else None,
    )
)

utility_data = [
    {
        "instruction": prompt,
        "output": response,
    } for prompt, response in zip(
        utility_prompts,
        utility_responses,
    )
]

with open(args.utility_save_path, "w") as f:
    json.dump(
        obj=utility_data,
        fp=f,
    )