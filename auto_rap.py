import argparse
import csv

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
)
from tqdm import tqdm

from utils import (
    load_tokenizer,
    get_tokenized_inputs,
)

DEFAULT_TOP_K = 20
DEFAULT_MAX_STEPS = 256

def find_subsequence_start(
    sequence: torch.Tensor,
    subsequence: torch.Tensor,
) -> int:
    """
    Finds the index of the first subsequence token in a sequence.

    Returns None if the subsequence token is not found.

    Args:
        sequence: The full sequence tensor.
        subsequence: The subsequence tensor to find in the sequence.
    """
    assert len(sequence) >= len(subsequence), \
        "The sequence must be at least as long as the subsequence."

    for i in range(
        0,
        len(sequence) - len(subsequence) + 1,
    ):
        if torch.equal(
            sequence[i:i + len(subsequence)],
            subsequence,
        ):
            return i
    
    return None

# Argument parser for command line arguments
parser = argparse.ArgumentParser(
    description=(
        "Training script for the automated Rank-Assisted Prefilling (RAP) "
        "attack model."
    ),
)

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="the HuggingFace Hub model ID or local path of the model to attack",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="the path to the prefilling dataset to use for attack evaluation",
)
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="the path to save the attack results CSV file",
)
parser.add_argument(
    "--helper",
    type=str,
    required=True,
    help="the HuggingFace Hub model ID or local path of the helper model",
)
parser.add_argument(
    "--samples",
    type=int,
    default=None,
    help="the number of samples in the dataset to evaluate",
)
parser.add_argument(
    "--skip",
    type=int,
    default=0,
    help="the number of samples in the dataset to skip from the start",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=DEFAULT_TOP_K,
    help=(
        "the number of top tokens to consider from the target model at each "
        "step"
    ),
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=DEFAULT_MAX_STEPS,
    help="the maximum number of attack steps per prompt",
)

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.model,
    device_map="auto",
)
tokenizer = load_tokenizer(
    model_name_or_path=args.model,
)

helper_model = AutoModelForTokenClassification.from_pretrained(
    pretrained_model_name_or_path=args.helper,
    device_map="auto",
)
helper_tokenizer = load_tokenizer(
    model_name_or_path=args.helper,
)

dataset = load_dataset(
    path="csv",
    data_files=args.dataset_path,
    split="train",
)

if args.samples is not None:
    dataset = dataset.take(
        n=args.samples,
    )

if args.skip > 0:
    dataset = dataset.skip(
        n=args.skip,
    )

save_file = open(
    file=args.output_path,
    mode="w",
)
save_writer = csv.DictWriter(
    f=save_file,
    fieldnames=[
        "prompt",
        "output",
        "tokens_generated",
        "backtracks",
        "oom",
        "total_steps",
    ],
)
save_writer.writeheader()

for data in tqdm(
    iterable=dataset,
    desc="Prompt",
    dynamic_ncols=True,
    leave=True,
    position=0,
    colour="green",
):
    generated = 0
    backtracks = 0
    steps = 0
    oom = False

    prompt, prefill = data["prompt"], data["prefill"]

    inputs = get_tokenized_inputs(
        prompts=[prompt],
        tokenizer=tokenizer,
        prefills=[prefill],
    )
    inputs_no_prefill = get_tokenized_inputs(
        prompts=[prompt],
        tokenizer=tokenizer,
    )

    prefill_start = find_subsequence_start(
        sequence=inputs["input_ids"][0],
        subsequence=inputs_no_prefill["input_ids"][0],
    ) + len(inputs_no_prefill["input_ids"][0])
    generation_start = inputs["input_ids"].shape[1]
    

    inputs = inputs.to(model.device)

    prior = None

    for _ in tqdm(
        iterable=range(args.max_steps),
        desc="Attack Step",
        dynamic_ncols=True,
        leave=False,
        position=1,
        colour="magenta",
    ):
        try:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                logits_to_keep=1,
            )
            logits = outputs.logits[0, -1]

            next_token_probs = torch.nn.functional.softmax(
                logits,
                dim=0,
            )

            top_k_tokens = torch.topk(
                input=next_token_probs,
                k=args.top_k,
            ).indices

            helper_prefills = inputs["input_ids"][:, prefill_start:].repeat(
                args.top_k,
                1,
            )
            helper_prefills = torch.cat(
                tensors=[
                    helper_prefills,
                    torch.unsqueeze(
                        input=top_k_tokens,
                        dim=1,
                    ),
                ],
                dim=-1,
            )
            helper_prefills = tokenizer.batch_decode(
                sequences=helper_prefills,
            )

            helper_inputs = get_tokenized_inputs(
                prompts=[prompt for _ in range(args.top_k)],
                tokenizer=helper_tokenizer,
                prefills=helper_prefills,
            ).to(helper_model.device)
            helper_logits = helper_model(
                input_ids=helper_inputs["input_ids"],
                attention_mask=helper_inputs["attention_mask"],
            ).logits
            helper_predictions = torch.argmax(
                helper_logits,
                dim=-1,
            )

            harm_tokens = [
                top_k_tokens[i].item() for i in range(args.top_k) \
                    if helper_predictions[i, -1].item() == 1
            ]

            if prior is not None:
                if prior in harm_tokens:
                    harm_tokens = harm_tokens[
                        harm_tokens.index(prior) + 1:
                    ]

                prior = None
            
            if len(harm_tokens) > 0:
                decoded_choice = tokenizer.decode(
                    harm_tokens[0],
                ).strip()
                decoded_prev = tokenizer.batch_decode(
                    sequences=inputs["input_ids"][0, -2:],
                )
                decoded_prev = [s.strip() for s in decoded_prev]

                while len(harm_tokens) > 0 and (
                    decoded_choice == decoded_prev[-1] and \
                        decoded_choice == decoded_prev[-2]
                ):
                    harm_tokens = harm_tokens[1:]

                    if len(harm_tokens) > 0:
                        decoded_choice = tokenizer.decode(
                            harm_tokens[0],
                        ).strip()
            chosen = None

            # Backtracking
            if len(harm_tokens) == 0:
                prior = inputs["input_ids"][0, -1].item()

                inputs["input_ids"] = inputs["input_ids"][:, :-1]
                inputs["attention_mask"] = inputs["attention_mask"][:, :-1]

                generated -= 1
                backtracks += 1
            else:
                chosen = harm_tokens[0]

                inputs["input_ids"] = torch.cat(
                    [
                        inputs["input_ids"],
                        torch.tensor(
                            data=[[chosen]],
                            dtype=inputs["input_ids"].dtype,
                            device = inputs["input_ids"].device,
                        ),
                    ],
                    dim=-1,
                )
                inputs["attention_mask"] = torch.cat(
                    [
                        inputs["attention_mask"],
                        torch.ones(
                            1,
                            1,
                            device=inputs["attention_mask"].device,
                        )
                    ],
                    dim=-1,
                )

                generated += 1
            
            steps += 1
            
            if generated < 0:
                break
        except torch.OutOfMemoryError as e:
            oom = True
            torch.cuda.empty_cache()

            break
    
    full_response = tokenizer.decode(
        inputs["input_ids"][0, prefill_start:],
    ).strip()

    save_writer.writerow(
        {
            "prompt": prompt,
            "output": full_response,
            "tokens_generated": generated,
            "backtracks": backtracks,
            "oom": oom,
            "total_steps": steps,
        }
    )
    save_file.flush()

save_file.close()