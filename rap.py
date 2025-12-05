import argparse
import csv
import time
import os

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

DEFAULT_SEED = 0
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
parser.add_argument(
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    help=(
        "the random seed to use for reproducibility",
    ),
)

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.model,
    device_map="auto",
)
tokenizer = load_tokenizer(
    model_name_or_path=args.model,
)

dataset = load_dataset(
    path="csv",
    data_files=args.dataset_path,
    split="train",
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
        "terminated",
        "total_steps",
        "time_taken",
    ],
)
save_writer.writeheader()

os.system("clear")

prompt_iter = tqdm(
    iterable=dataset,
    desc="Prompt",
    dynamic_ncols=True,
    leave=False,
    position=0,
    colour="green",
)

for i, data in enumerate(prompt_iter):
    generated = 0
    backtracks = 0
    steps = 0
    oom = False
    terminated = False

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

    start_time = None
    end_time = None

    step_iterable = tqdm(
        iterable=range(args.max_steps),
        desc="Attack Step",
        dynamic_ncols=True,
        leave=False,
        position=1,
        colour="magenta",
    )
    
    for _ in step_iterable:
        backtrack = False
        user_quit = False
        chosen = None

        prefill = tokenizer.decode(
            inputs["input_ids"][0, prefill_start:],
        )

        print('\n')
        print('=' * 80)
        print("PROMPT:", repr(prompt))
        print('-' * 80)
        print('-' * 80)
        print("PREFILL:", repr(prefill))
        print('-' * 80)

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

            decoded_top_k_tokens = tokenizer.batch_decode(
                sequences=top_k_tokens,
            )

            options = ""

            for j, token in enumerate(decoded_top_k_tokens):
                options += f"{j + 1}) " + repr(token) + "\n"
            
            print(options)
            
            action = input("> ")
            valid_action = False

            if start_time is None:
                start_time = time.perf_counter()

            while not valid_action:
                if action.strip().lower() == "b":
                    backtrack = True
                    valid_action = True
                elif action.strip().lower() == "s":
                    terminated = True
                    valid_action = True
                elif action.strip().lower() == "q":
                    user_quit = True
                    valid_action = True
                else:
                    try:
                        choice = int(action)

                        if choice >= 1 and choice <= len(decoded_top_k_tokens):
                            chosen = top_k_tokens[choice - 1].unsqueeze(0)
                            valid_action = True
                    except ValueError:
                        pass
                
                if not valid_action:
                    action = input("> ")
            
            if backtrack:
                inputs["input_ids"] = inputs["input_ids"][:, :-1]
                inputs["attention_mask"] = inputs["attention_mask"][:, :-1]

                generated -= 1
                backtracks += 1
            elif not terminated and not user_quit:
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
            
            if generated < 0 or terminated or user_quit:
                break
        except torch.OutOfMemoryError as e:
            oom = True

            torch.cuda.empty_cache()

            break
    
        os.system("clear")

        prompt_iter.refresh()
        step_iterable.refresh()
    
    end_time = time.perf_counter()
    
    full_response = tokenizer.decode(
        inputs["input_ids"][0, prefill_start:],
    ).strip()
    total_time = end_time - start_time

    save_writer.writerow(
        {
            "prompt": prompt,
            "output": full_response,
            "tokens_generated": generated,
            "backtracks": backtracks,
            "oom": oom,
            "terminated": terminated,
            "total_steps": steps,
            "time_taken": total_time,
        }
    )
    save_file.flush()

    if user_quit:
        break

save_file.close()