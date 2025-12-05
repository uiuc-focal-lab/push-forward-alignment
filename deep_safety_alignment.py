import argparse
import random
from math import ceil
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    BatchEncoding,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from accelerate.utils import set_seed
from deepspeed import DeepSpeedEngine
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (
    load_tokenizer,
    get_tokenized_inputs,
    get_log_perplexity,
)

# Default values for training parameters
DEFAULT_SEED = 0
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_MAX_PREFILL = 100
DEFAULT_SAFETY_BATCH_SIZE = 4
DEFAULT_UTILITY_BATCH_SIZE = 16
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# Weights & Biases
WANDB_PROJECT_NAME = "llm-prefilling"

# Error messages
PREFILL_INDEX_NOT_FOUND_ERROR = (
    "A prefill token sequence could not be completed isolated from the other "
    "tokens in the input. This is likely due to tokenization effects at the "
    "boundary."
)
ALL_MASKED_OUT_ERROR = (
    "All sequences in the batch were masked out. Consider increasing the "
    "maximum sequence length."
)

CHECKPOINT_NAME = "checkpoint"

def get_prepared_batch(
    safety_batch: dict,
    utility_batch: dict,
    tokenizer: PreTrainedTokenizer,
    max_prefill: int,
    max_seq_len: int,
    epoch: int,
    grad_step: int,
    seed: int,
    device: torch.device,
    system_prompt: str = None,
) -> tuple[
    BatchEncoding,
    list[int],
    list[int],
    list[int],
    BatchEncoding,
    torch.Tensor,
    torch.Tensor,
]:
    """Returns prepared training data for the current batch.

    Args:
        safety_batch: A dictionary containing the safety data.
        utility_batch: A dictionary containing the utility data.
        tokenizer: The tokenizer to use for encoding the inputs.
        max_prefill: The maximum number of tokens to prefill for the safety
            objective.
        max_seq_len: The maximum sequence length for the model.
        epoch: The current training epoch. Used to set the random seed.
        grad_step: The current training gradient step. Used to set the random 
            seed.
        seed: The base random seed for reproducibility.
        device: The device to which the inputs should be moved.
        system_prompt: An optional system prompt.
    """

    def find_prefill_index(
        input_full: torch.Tensor,
        input_no_prefill: torch.Tensor,
    ) -> int:
        """
        Finds the index of the first prefill token in the full input.

        Returns None if the prefill token is not found.

        Args:
            input_full: The full input tensor with prefill tokens.
            input_no_prefill: The input tensor without prefill tokens.
        """
        for i in range(
            0,
            len(input_full) - len(input_no_prefill) + 1,
        ):
            if torch.equal(
                input_full[i:i + len(input_no_prefill)],
                input_no_prefill,
            ):
                return i + len(input_no_prefill)
        
        return None
    
    safety_inputs = get_tokenized_inputs(
        prompts=safety_batch["instruction"],
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        prefills=safety_batch["harmful"],
        max_length=max_seq_len,
    ).to(device)
    safety_inputs_no_prefill = get_tokenized_inputs(
        prompts=safety_batch["instruction"],
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_length=max_seq_len,
    ).to(device)
    utility_inputs = get_tokenized_inputs(
        prompts=utility_batch["instruction"],
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        prefills=[s + tokenizer.eos_token for s in utility_batch["output"]],
        max_length=max_seq_len,
    ).to(device)
    utility_inputs_no_prefill = get_tokenized_inputs(
        prompts=utility_batch["instruction"],
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_length=max_seq_len,
    ).to(device)

    first_non_pad = []
    first_prefill = []
    first_refusal = []

    num_instructions = len(safety_batch["instruction"])

    # Set safety seed for consistency across different total number of processes
    safety_seed = \
        "seed_" + str(seed) + "_epoch_" + str(epoch) + "_step_" + str(grad_step)

    refusals = safety_batch["refusal"]

    all_non_prefill_tokens = []
    all_prefill_tokens = []
    all_refusal_tokens = []
    all_total_lengths = []

    # Construct safety inputs and loss mask
    for i in range(len(safety_inputs["input_ids"])):
        prompt_safety_seed = safety_seed + "_index_" + str(
            safety_batch["index"][i]
        )

        random.seed(prompt_safety_seed)

        input_no_prefill = safety_inputs_no_prefill["input_ids"][i]
        input_full = safety_inputs["input_ids"][i]

        # Remove padding from the no-prefill input
        no_prefill_non_pad_index = \
            (
                input_no_prefill != tokenizer.pad_token_id
            ).nonzero().squeeze()[0].item()

        input_no_prefill = input_no_prefill[no_prefill_non_pad_index:]

        # Find index of the first prefill token
        init_prefill_index = find_prefill_index(
            input_full=input_full,
            input_no_prefill=input_no_prefill,
        )
        
        if init_prefill_index is None:
            raise ValueError(PREFILL_INDEX_NOT_FOUND_ERROR)
        
        prefill_tokens = input_full[init_prefill_index:]

        coin_flip = random.randrange(2)
        add_prefill = coin_flip == 1
        
        # Find index of the first non-padding token
        non_pad_index = \
            (input_full != tokenizer.pad_token_id).nonzero().squeeze()[0].item()
        
        non_prefill_tokens = input_full[non_pad_index:init_prefill_index]

        refusal_tokens = tokenizer(
            text=(refusals[i] + tokenizer.eos_token),
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt",
        ).to(device)["input_ids"][0]

        # With 50% probability, add prefill before refusal
        if add_prefill:
            prefill_length = random.randrange(
                start=1,
                stop=min(
                    len(prefill_tokens),
                    max_prefill,
                ),
            )
            prefill_tokens = prefill_tokens[:prefill_length]
        else:
            # With 50% probability, full refusal and no prefill
            prefill_tokens = prefill_tokens[:0]
        
        total_length = \
            len(non_prefill_tokens) + len(prefill_tokens) + len(refusal_tokens)
        total_length = min(total_length, max_seq_len)

        all_total_lengths.append(total_length)
        
        all_non_prefill_tokens.append(
            non_prefill_tokens[:max_seq_len]
        )
        all_prefill_tokens.append(
            prefill_tokens[
                :max(
                    0,
                    max_seq_len - len(non_prefill_tokens),
                )
            ]
        )
        all_refusal_tokens.append(
            refusal_tokens[
                :max(
                    0,
                    max_seq_len - len(non_prefill_tokens) - len(prefill_tokens),
                )
            ]
        )
    
    max_augmented_len = max(all_total_lengths)
    safety_inputs["input_ids"] = torch.full(
        size=(
            len(safety_inputs["input_ids"]),
            max_augmented_len,
        ),
        fill_value=tokenizer.pad_token_id,
        dtype=safety_inputs["input_ids"].dtype,
        device=device,
    )
    safety_inputs["attention_mask"] = torch.zeros(
        size=(
            len(safety_inputs["input_ids"]),
            max_augmented_len,
        ),
        dtype=safety_inputs["attention_mask"].dtype,
        device=device,
    )
    safety_loss_mask = safety_inputs["attention_mask"].clone()

    for i, (
        non_prefill_tokens,
        prefill_tokens,
        refusal_tokens,
        total_length,
    ) in enumerate(
        zip(
            all_non_prefill_tokens,
            all_prefill_tokens,
            all_refusal_tokens,
            all_total_lengths,
        )
    ):
        num_pad_tokens = max_augmented_len - total_length
        prefill_index = num_pad_tokens + len(non_prefill_tokens)
        refusal_index = prefill_index + len(prefill_tokens)

        first_non_pad.append(num_pad_tokens)
        first_prefill.append(prefill_index)
        first_refusal.append(refusal_index)

        safety_inputs["input_ids"][i, num_pad_tokens:prefill_index] = \
            non_prefill_tokens
        safety_inputs["input_ids"][i, prefill_index:refusal_index] = \
            prefill_tokens
        safety_inputs["input_ids"][i, refusal_index:] = refusal_tokens

        safety_inputs["attention_mask"][i, num_pad_tokens:] = 1

        """When a prefill is added, both the prefill and refusal tokens are 
        involved in the loss; otherwise, its just the refusal tokens."""
        safety_loss_mask[i, prefill_index:] = 1

    # Construct utility loss mask
    no_prefill_non_pad_lengths = (
        utility_inputs_no_prefill["input_ids"] != tokenizer.pad_token_id
    ).sum(dim=1)
    non_pad_index = torch.argmax(
        input=(
            utility_inputs["input_ids"] != tokenizer.pad_token_id
        ).long(),
        dim=1,
    )

    prefill_indices = non_pad_index + no_prefill_non_pad_lengths

    utility_loss_mask = torch.arange(
        start=0,
        end=utility_inputs["input_ids"].shape[1],
        device=device,
    ).unsqueeze(0) >= prefill_indices.unsqueeze(1)
    
    # Remove sequences that are entirely masked out
    for i, (inputs, loss_mask) in enumerate([
        (safety_inputs, safety_loss_mask),
        (utility_inputs, utility_loss_mask),
    ]):
        valid_indices = loss_mask.any(dim=1).cpu()
        inputs["input_ids"] = inputs["input_ids"][valid_indices]
        inputs["attention_mask"] = inputs["attention_mask"][valid_indices]

        if i == 0: # Safety inputs
            safety_loss_mask = safety_loss_mask[valid_indices]

            first_non_pad = [
                first_non_pad[j] for j in range(
                    len(first_non_pad)
                ) if valid_indices[j]
            ]
            first_prefill = [
                first_prefill[j] for j in range(
                    len(first_prefill)
                ) if valid_indices[j]
            ]
            first_refusal = [
                first_refusal[j] for j in range(
                    len(first_refusal)
                ) if valid_indices[j]
            ]
        else: # Utility inputs
            utility_loss_mask = utility_loss_mask[valid_indices]

    if len(safety_loss_mask) == 0 and len(utility_loss_mask) == 0:
        raise ValueError(ALL_MASKED_OUT_ERROR)

    return (
        safety_inputs,
        first_non_pad,
        first_prefill,
        first_refusal,
        utility_inputs,
        safety_loss_mask,
        utility_loss_mask,
    )

def collate_safety_batch(
    batch: list[dict],
) -> dict:
    """Collates a batch of safety data into a single dictionary.

    Args:
        batch: A list containing each item in the batch. Each item is a 
        dictionary containing the safety data.

    Returns:
        A dictionary containing the collated safety data.
    """

    collated_batch = {
        "instruction": [],
        "harmful": [],
        "refusal": [],
        "index": [],
    }

    for i, data in enumerate(batch):
        collated_batch["instruction"].append(data["instruction"])
        collated_batch["harmful"].append(data["harmful"])
        collated_batch["refusal"].append(data["refusal"])
        collated_batch["index"].append(data["index"])
    
    return collated_batch

def find_first_non_pad_in_batch(
    attention_mask: torch.Tensor,
) -> int:
    """
    Returns the first column index containing a non-padding token in the batch.

    Args:
        attention_mask: The attention mask tensor.
    """

    for i in range(attention_mask.shape[1]):
        if attention_mask[:, i].any():
            return i
    
    return None

def save_model(
    model: DeepSpeedEngine,
    unwrapped_model: PreTrainedModel,
    save_dir: str,
    save_function: callable,
    zero_stage: int,
    is_main_process: bool,
):
    """Saves the model to the specified directory.
    
    Args:
        model: The DeepSpeed wrapped model.
        unwrapped_model: The unwrapped model.
        save_dir: The directory to save the model to.
        save_function: The function to use for saving the model.
        zero_stage: The DeepSpeed ZeRO optimization stage.
        is_main_process: Whether the current process is the main process.
    """

    if zero_stage == 3:
        model.save_checkpoint(
            save_dir,
            CHECKPOINT_NAME,
        )

        if is_main_process:
            unwrapped_model.config.save_pretrained(
                save_directory=save_dir,
            )
    else:
        unwrapped_model.save_pretrained(
            save_directory=save_dir,
            is_main_process=is_main_process,
            save_function=save_function,
        )

# Argument parser for command line arguments
parser = argparse.ArgumentParser(
    description="Attention-regularized deep alignment fine-tuning script.",
)

# Model and dataset arguments
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="the HuggingFace Hub model ID or local path of the model to fine-tune",
)
parser.add_argument(
    "--safety_dataset_path",
    type=str,
    required=True,
    help="the path to the dataset used for the safety objective",
)
parser.add_argument(
    "--utility_dataset_path",
    type=str,
    required=True,
    help="the path to the dataset used for the utility objective",
)

# Training arguments
parser.add_argument(
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    help="the random seed for reproducibility",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=DEFAULT_EPOCHS,
    help="the number of epochs to train the model",
)
parser.add_argument(
    "--system_prompt",
    action="store_true",
    help="add a system prompt that encourages safety",
)
parser.add_argument(
    "--presto",
    action="store_true",
    help="use PRESTO regularization",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=DEFAULT_LEARNING_RATE,
    help="the learning rate for the optimizer",
)
parser.add_argument(
    "--max_seq_len",
    type=int,
    default=DEFAULT_MAX_SEQ_LEN,
    help="the maximum sequence length for fine-tuning the model",
)
parser.add_argument(
    "--max_prefill",
    type=int,
    default=DEFAULT_MAX_PREFILL,
    help="the maximum number of tokens to prefill for the safety objective",
)
parser.add_argument(
    "--safety_batch_size_per_device",
    type=int,
    default=DEFAULT_SAFETY_BATCH_SIZE,
    help="the amount of safety data to use in each batch per device",
)
parser.add_argument(
    "--utility_batch_size_per_device",
    type=int,
    default=DEFAULT_UTILITY_BATCH_SIZE,
    help="the amount of utility data to use in each batch per device",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    help="the number of gradient accumulation steps to use",
)
parser.add_argument(
    "--save_dir",
    type=str,
    required=True,
    help="the directory to save the fine-tuned model",
)
parser.add_argument(
    "--show_batch_tqdm",
    action="store_true",
    help="whether to show a tqdm progress bar for each batch",
)
parser.add_argument(
    "--wandb_run_name",
    type=str,
    default=None,
    help="the name of the Weights & Biases run",
)

args = parser.parse_args()

set_seed(args.seed)

accelerator = Accelerator(
    log_with="wandb",
)
device = accelerator.device

accelerator.init_trackers(
    project_name=WANDB_PROJECT_NAME,
    config={
        "model": args.model,
        "safety_dataset_path": args.safety_dataset_path,
        "utility_dataset_path": args.utility_dataset_path,
        "seed": args.seed,
        "epochs": args.epochs,
        "system_prompt": args.system_prompt,
        "presto": args.presto,
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len,
        "max_prefill": args.max_prefill,
        "safety_batch_size_per_device": args.safety_batch_size_per_device,
        "utility_batch_size_per_device": args.utility_batch_size_per_device,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_dir": args.save_dir,
    },
    init_kwargs={
        "wandb": {
            "name": (
                args.wandb_run_name if args.wandb_run_name is not None else \
                    args.model
            ),
        }
    },
)

if accelerator.state.deepspeed_plugin is not None:
    zero_stage = accelerator.state.deepspeed_plugin.deepspeed_config.get(
        "zero_optimization",
        {},
    ).get(
        "stage",
        None,
    )
else:
    zero_stage = None

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.model,
    **({"attn_implementation": "eager"} if args.presto else {}),
).train()
tokenizer = load_tokenizer(
    model_name_or_path=args.model,
)

safety_dataset = load_dataset(
    path="json",
    data_files=args.safety_dataset_path,
    split="train",
)
safety_dataset = safety_dataset.add_column(
    name="index", 
    column=range(len(safety_dataset)),
)

utility_dataset = load_dataset(
    path="json",
    data_files=args.utility_dataset_path,
    split="train",
)

safety_dataloader = DataLoader(
    dataset=safety_dataset,
    batch_size=args.safety_batch_size_per_device,
    shuffle=True,
    collate_fn=collate_safety_batch,
)

utility_dataloader = DataLoader(
    dataset=utility_dataset,
    batch_size=args.utility_batch_size_per_device,
    shuffle=True,
)

optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=args.learning_rate,
)

model, optimizer, safety_dataloader, utility_dataloader = accelerator.prepare(
    model,
    optimizer,
    safety_dataloader,
    utility_dataloader,
)

safety_iterator = iter(safety_dataloader)
utility_iterator = iter(utility_dataloader)

total_grad_steps = ceil(
    len(safety_dataloader) / args.gradient_accumulation_steps
)

scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(
        total_grad_steps * accelerator.num_processes * args.epochs
    ),
)
scheduler = accelerator.prepare_scheduler(scheduler)

final_batch_size = len(safety_dataloader) % args.gradient_accumulation_steps
final_batch_size = final_batch_size if final_batch_size > 0 else \
    args.gradient_accumulation_steps

epoch_iterable = range(args.epochs)

if accelerator.is_main_process:
    tokenizer.save_pretrained(
        save_directory=args.save_dir,
    )

    epoch_iterable = tqdm(
        iterable=epoch_iterable,
        desc="Epochs",
        dynamic_ncols=True,
        position=0,
        colour="green",
    )

for epoch in epoch_iterable:
    grad_step_iterable = range(total_grad_steps)

    if accelerator.is_main_process:
        grad_step_iterable = tqdm(
            iterable=grad_step_iterable,
            desc="Step",
            dynamic_ncols=True,
            position=1,
            leave=False,
            colour="magenta",
        )

    for grad_step in grad_step_iterable:
        optimizer.zero_grad()

        last_step = (grad_step == (total_grad_steps - 1))
        batches_in_step = final_batch_size if last_step else \
            args.gradient_accumulation_steps
        
        batch_iterable = []
        local_token_count = torch.tensor(0, device=device)

        # Collect all batches for the current gradient step
        for _ in range(batches_in_step):
            safety_batch = next(safety_iterator)

            try:
                utility_batch = next(utility_iterator)
            except StopIteration:
                # If the utility iterator is exhausted, reset it
                utility_iterator = iter(utility_dataloader)
                utility_batch = next(utility_iterator)
            
            (
                safety_inputs,
                first_non_pad,
                first_prefill,
                first_refusal,
                utility_inputs,
                safety_loss_mask,
                utility_loss_mask,
            ) = get_prepared_batch(
                safety_batch=safety_batch,
                utility_batch=utility_batch,
                tokenizer=tokenizer,
                max_prefill=args.max_prefill,
                max_seq_len=args.max_seq_len,
                epoch=epoch,
                grad_step=grad_step,
                seed=args.seed,
                device=device,
                system_prompt=SYSTEM_PROMPT if args.system_prompt else None,
            )

            has_prefill = [
                r > p for p, r in zip(
                    first_prefill,
                    first_refusal,
                )
            ]
            no_prefill = [
                not r for r in has_prefill
            ]
            local_token_count += safety_loss_mask.sum() + \
                utility_loss_mask.sum()

            batch_iterable.append(
                (
                    safety_inputs,
                    first_non_pad,
                    first_prefill,
                    first_refusal,
                    has_prefill,
                    no_prefill,
                    utility_inputs,
                    safety_loss_mask,
                    utility_loss_mask,
                )
            )
        
        # Calculate total number of loss-contributing tokens
        total_token_count = accelerator.reduce(
            tensor=local_token_count,
            reduction="sum",
        )

        scale_factor = accelerator.num_processes * batches_in_step / \
            total_token_count

        if (
            accelerator.is_main_process and \
                args.gradient_accumulation_steps > 1 and args.show_batch_tqdm
        ):
            batch_iterable = tqdm(
                iterable=batch_iterable,
                desc="Batch",
                dynamic_ncols=True,
                position=2,
                leave=False,
                colour="white",
            )

        # Initialize accumulated losses for logging
        local_accumulated_loss = torch.zeros(
            1,
            device=device,
        )
        local_accumulated_safety_loss = torch.zeros(
            1,
            device=device,
        )
        local_accumulated_safety_no_prefill_loss = torch.zeros(
            1,
            device=device,
        )
        local_accumulated_safety_prefill_loss = torch.zeros(
            1,
            device=device,
        )
        local_accumulated_safety_prefill_attention_loss = torch.zeros(
            1,
            device=device,
        )
        local_accumulated_safety_prefill_refusal_loss = torch.zeros(
            1,
            device=device,
        )
        local_accumulated_utility_loss = torch.zeros(
            1,
            device=device,
        )
        
        # Compute accumulated loss and gradients
        for i, (
            safety_inputs,
            first_non_pad,
            first_prefill,
            first_refusal,
            has_prefill,
            no_prefill,
            utility_inputs,
            safety_loss_mask,
            utility_loss_mask,
        ) in enumerate(batch_iterable):
            num_safety_inputs = len(safety_inputs["input_ids"])
            num_utility_inputs = len(utility_inputs["input_ids"])

            safety_additional_pad = max(
                0,
                utility_inputs["input_ids"].size(1) - safety_inputs["input_ids"].size(1)
            )
            utility_additional_pad = max(
                0,
                safety_inputs["input_ids"].size(1) - utility_inputs["input_ids"].size(1)
            )

            safety_inputs["input_ids"] = torch.nn.functional.pad(
                input=safety_inputs["input_ids"],
                pad=(safety_additional_pad, 0),
                value=tokenizer.pad_token_id,
            )
            safety_inputs["attention_mask"] = torch.nn.functional.pad(
                input=safety_inputs["attention_mask"],
                pad=(safety_additional_pad, 0),
            )

            utility_inputs["input_ids"] = torch.nn.functional.pad(
                input=utility_inputs["input_ids"],
                pad=(utility_additional_pad, 0),
                value=tokenizer.pad_token_id,
            )
            utility_inputs["attention_mask"] = torch.nn.functional.pad(
                input=utility_inputs["attention_mask"],
                pad=(utility_additional_pad, 0),
            )

            input_ids = torch.cat(
                tensors=[
                    safety_inputs["input_ids"],
                    utility_inputs["input_ids"],
                ],
                dim=0,
            )
            attention_mask = torch.cat(
                tensors=[
                    safety_inputs["attention_mask"],
                    utility_inputs["attention_mask"],
                ],
                dim=0,
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_attentions=args.presto,
            )

            safety_logits = outputs.logits[
                :num_safety_inputs,
                safety_additional_pad:
            ]
            utility_logits = outputs.logits[
                num_safety_inputs:,
                utility_additional_pad:
            ]

            safety_inputs["input_ids"] = safety_inputs["input_ids"][
                :,
                safety_additional_pad:
            ]
            utility_inputs["input_ids"] = utility_inputs["input_ids"][
                :,
                utility_additional_pad:
            ]

            safety_no_prefill_loss = get_log_perplexity(
                inputs=safety_inputs["input_ids"][no_prefill],
                logits=safety_logits[no_prefill],
                mask=safety_loss_mask[no_prefill],
                reduction="batch_sum",
            ) * scale_factor if sum(no_prefill) > 0 else torch.zeros(
                1,
                device=device,
                dtype=safety_logits.dtype,
            )

            safety_prefill_attention_loss = torch.zeros(
                1,
                device=device,
                dtype=safety_logits.dtype,
            )
            safety_prefill_refusal_loss = torch.zeros(
                1,
                device=device,
                dtype=safety_logits.dtype,
            )

            if sum(has_prefill) > 0:
                first_refusal = [
                    token for token, p in zip(
                        first_refusal,
                        has_prefill,
                    ) if p
                ]

                if args.presto:
                    first_non_pad = [
                        token for token, p in zip(
                            first_non_pad,
                            has_prefill,
                        ) if p
                    ]
                    first_prefill = [
                        token for token, p in zip(
                            first_prefill,
                            has_prefill,
                        ) if p
                    ]

                    attention_loss_scale_factor = scale_factor / \
                        len(outputs.attentions)

                    # Attention regularization
                    for layer_attentions in outputs.attentions:
                        layer_attentions = layer_attentions[
                            :num_safety_inputs
                        ][has_prefill]

                        for j in range(sum(has_prefill)):
                            response_length = layer_attentions.shape[-1] - \
                                first_prefill[j]
                            prefill_length = first_refusal[j] - \
                                first_prefill[j]

                            for response_token in range(response_length):
                                prefill_sum_end = first_prefill[j] + \
                                    min(
                                        prefill_length,
                                        response_token + 1,
                                    )
                                prefill_sum = layer_attentions[
                                    j,
                                    :,
                                    first_prefill[j] + response_token,
                                    first_prefill[j]:prefill_sum_end,
                                ].sum()

                                before_prefill_sum = layer_attentions[
                                    j,
                                    :,
                                    first_prefill[j] + response_token,
                                    first_non_pad[j]:first_prefill[j],
                                ].sum()

                                after_prefill_sum_end = \
                                    first_prefill[j] + response_token + 1
                                after_prefill_sum = layer_attentions[
                                    j,
                                    :,
                                    first_prefill[j] + response_token,
                                    first_refusal[j]:after_prefill_sum_end,
                                ].sum()

                                safety_prefill_attention_loss += (
                                    prefill_sum - (
                                        before_prefill_sum + after_prefill_sum
                                    ) + layer_attentions.shape[1] # min loss = 0
                                ) * attention_loss_scale_factor
                
                prefill_safety_loss_mask = safety_loss_mask[has_prefill].clone()

                for j in range(sum(has_prefill)):
                    prefill_safety_loss_mask[
                        j, :first_refusal[j]
                    ] = False
                
                safety_prefill_refusal_loss = get_log_perplexity(
                    inputs=safety_inputs["input_ids"][has_prefill],
                    logits=safety_logits[has_prefill],
                    mask=prefill_safety_loss_mask,
                    reduction="batch_sum",
                ) * scale_factor

            safety_prefill_loss = safety_prefill_attention_loss + \
                safety_prefill_refusal_loss
            safety_loss = safety_no_prefill_loss + safety_prefill_loss

            utility_loss = get_log_perplexity(
                inputs=utility_inputs["input_ids"],
                logits=utility_logits,
                mask=utility_loss_mask,
                reduction="batch_sum",
            ) * scale_factor if num_utility_inputs > 0 else torch.zeros(
                1,
                device=device,
                dtype=utility_logits.dtype,
            )

            loss = safety_loss + utility_loss

            local_accumulated_loss += loss.detach()
            local_accumulated_safety_loss += safety_loss.detach()
            local_accumulated_safety_no_prefill_loss += \
                safety_no_prefill_loss.detach()
            local_accumulated_safety_prefill_loss += \
                safety_prefill_loss.detach()
            local_accumulated_safety_prefill_refusal_loss += \
                safety_prefill_refusal_loss.detach()
            local_accumulated_utility_loss += utility_loss.detach()

            if args.presto:
                local_accumulated_safety_prefill_attention_loss += \
                    safety_prefill_attention_loss.detach()

            accelerator.backward(loss)

        # Accumuate losses for logging
        total_accumulated_loss = accelerator.reduce(
            tensor=local_accumulated_loss,
            reduction="mean",
        ) / batches_in_step
        total_accumulated_safety_loss = accelerator.reduce(
            tensor=local_accumulated_safety_loss,
            reduction="mean",
        ) / batches_in_step
        total_accumulated_safety_no_prefill_loss = accelerator.reduce(
            tensor=local_accumulated_safety_no_prefill_loss,
            reduction="mean",
        ) / batches_in_step
        total_accumulated_safety_prefill_loss = accelerator.reduce(
            tensor=local_accumulated_safety_prefill_loss,
            reduction="mean",
        ) / batches_in_step
        total_accumulated_safety_prefill_refusal_loss = accelerator.reduce(
            tensor=local_accumulated_safety_prefill_refusal_loss,
            reduction="mean",
        ) / batches_in_step
        total_accumulated_utility_loss = accelerator.reduce(
            tensor=local_accumulated_utility_loss,
            reduction="mean",
        ) / batches_in_step

        if args.presto:
            total_accumulated_safety_prefill_attention_loss = \
                accelerator.reduce(
                    tensor=local_accumulated_safety_prefill_attention_loss,
                    reduction="mean",
                ) / batches_in_step

        current_lr = optimizer.state_dict().get(
            "optimizer_state_dict" if zero_stage == 3 else \
                "base_optimizer_state",
            {},
        ).get(
            "param_groups",
            [{}],
        )[0].get(
            "lr",
            None,
        )

        # Log training information
        log_data = {
            "train_loss": total_accumulated_loss.item(),
            "train_safety_loss": total_accumulated_safety_loss.item(),
            "train_safety_no_prefill_loss": \
                total_accumulated_safety_no_prefill_loss.item(),
            "train_safety_prefill_loss": \
                total_accumulated_safety_prefill_loss.item(),
            "train_utility_loss": total_accumulated_utility_loss.item(),
            "lr": current_lr,
        }

        if args.presto:
            log_data["train_safety_prefill_attention_loss"] = \
                total_accumulated_safety_prefill_attention_loss.item()
            log_data["train_safety_prefill_refusal_loss"] = \
                total_accumulated_safety_prefill_refusal_loss.item()
        
        accelerator.log(log_data)

        if accelerator.is_main_process:
            grad_step_iterable.set_postfix({
                "loss": total_accumulated_loss.item(),
            })

        optimizer.step()
        scheduler.step()

    # Safety iterator is exhausted, reset it
    safety_iterator = iter(safety_dataloader)

    # Save model at the end of each epoch
    accelerator.wait_for_everyone()

    save_model(
        model=model,
        unwrapped_model=accelerator.unwrap_model(model),
        save_dir=args.save_dir,
        save_function=accelerator.save,
        zero_stage=zero_stage,
        is_main_process=accelerator.is_main_process,
    )

accelerator.end_training()
