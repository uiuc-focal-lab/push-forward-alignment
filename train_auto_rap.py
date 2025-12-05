import argparse
import random
from math import ceil
from pathlib import Path

import torch
from transformers import (
    AutoModelForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    BatchEncoding,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from accelerate.utils import set_seed
from deepspeed import DeepSpeedEngine
from datasets import load_from_disk
from torch.utils.data import DataLoader
import nltk
from nltk.tokenize import PunktTokenizer
from tqdm import tqdm

from utils import (
    load_tokenizer,
    get_tokenized_inputs,
)

IGNORE_INDEX = -100

# Default values for training parameters
DEFAULT_SEED = 0
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_MAX_PREFILL = 100
DEFAULT_BATCH_SIZE = 1
DEFAULT_VAL_SIZE = 0.2
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1

# Weights & Biases
WANDB_PROJECT_NAME = "llm-prefilling"

CHECKPOINT_NAME = "checkpoint"

def get_prepared_batch(
    batch: dict,
    tokenizer: PreTrainedTokenizer,
    target_tokenizer: PreTrainedTokenizer,
    punkt_tokenizer: PunktTokenizer,
    max_prefill: int,
    epoch: int,
    grad_step: int,
    seed: int,
    device: torch.device,
) -> tuple[
    BatchEncoding,
    torch.tensor,
]:
    """Returns prepared training data for the current batch.

    Args:
        batch: A dictionary containing the batch data.
        tokenizer: The tokenizer to use for encoding the inputs.
        target_tokenizer: The tokenizer of the target model.
        punkt_tokenizer: The Punkt tokenizer for sentence tokenization.
        max_prefill: The maximum number of tokens to prefill for the safety
            objective.
        epoch: The current training epoch. Used to set the random seed.
        grad_step: The current training gradient step. Used to set the random 
            seed.
        seed: The base random seed for reproducibility.
        device: The device to which the inputs should be moved.
    """
    
    def punctuation_cut(
        prefill: str,
    ) -> str:
        """Cuts the prefill string after a random punctuation mark.

        Args:
            prefill: The prefill string to cut.
        """

        punkt_tokenized = punkt_tokenizer.tokenize(prefill)
        start_indices = [
            prefill.index(tokens) for tokens in punkt_tokenized
        ]
        final_tokenized = []

        for i in range(len(start_indices)):
            if i < len(start_indices) - 1:
                final_tokenized.append(
                    prefill[start_indices[i]:start_indices[i + 1]]
                )
            else:
                final_tokenized.append(
                    prefill[start_indices[i]:]
                )
        
        cutoff_index = random.randrange(
            start=1,
            stop=len(final_tokenized) + 1,
        )
        final_tokenized = final_tokenized[:cutoff_index]

        prefill = "".join(final_tokenized).rstrip()

        return prefill
    
    harmful_responses = batch["harmful_response"]

    safe_responses = [
        batch["safe_response"][
            random.randrange(
                start=0,
                stop=len(batch["safe_response"]),
            )
        ][i] for i in range(len(batch["prompt"]))
    ]

    split_harmful_punctuation = [
        random.random() < 0.5 for _ in range(len(harmful_responses))
    ]
    harmful_responses = [
        punctuation_cut(harmful_responses[i]) if split_harmful_punctuation[i] \
            else harmful_responses[i] for i in range(len(harmful_responses))
    ]

    harmful_target_tokens = target_tokenizer(
        text=harmful_responses,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    safe_target_tokens = target_tokenizer(
        text=safe_responses,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    prefills = []
    labels = []

    # Set seed for consistency across different total number of processes
    seed = str(seed) + str(epoch) + str(grad_step)

    for i in batch["index"]:
        seed += str(i)
    
    random.seed(seed)

    for split, harmful, safe in zip(
        split_harmful_punctuation,
        harmful_target_tokens,
        safe_target_tokens,
    ):
        add_safe = random.random() < 0.5

        if not split:
            harmful_length = random.randrange(
                start=min(
                    10,
                    max_prefill,
                    len(harmful),
                ),
                stop=min(
                    max_prefill,
                    len(harmful),
                ) + 1,
            )

            harmful = harmful[:harmful_length]

        if add_safe and len(safe) > 0:
            safe_length = random.randrange(
                start=1,
                stop=min(
                    5,
                    len(safe),
                ) + 1,
            )
        else:
            safe_length = 0
        
        safe = safe[:safe_length]

        prefill = target_tokenizer.decode(
            token_ids=harmful + safe,
        )
        label = 1 - int(add_safe)

        prefills.append(prefill)
        labels.append(label)
    
    inputs = get_tokenized_inputs(
        prompts=batch["prompt"],
        tokenizer=tokenizer,
        prefills=prefills,
    ).to(device)

    loss_targets = torch.empty(
        size=inputs["input_ids"].shape,
        dtype=torch.long,
        device=device,
    ).fill_(IGNORE_INDEX)
    loss_targets[:, -1] = torch.tensor(
        labels,
        dtype=torch.long,
        device=device,
    )

    return (
        inputs,
        loss_targets,
    )

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

def get_args() -> argparse.Namespace:
    """Parses command line arguments."""

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
        help="the HuggingFace Hub model ID or local path of the model to fine-tune",
    )
    parser.add_argument(
        "--target_model",
        type=str,
        required=True,
        help="the HuggingFace Hub model ID or local path of the target model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="the dataset to use",
    )
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
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="the learning rate for the optimizer",
    )
    parser.add_argument(
        "--max_prefill",
        type=int,
        default=DEFAULT_MAX_PREFILL,
        help="the maximum number of harmful tokens to prefill",
    )
    parser.add_argument(
        "--batch_size_per_device",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="the batch size per device",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=DEFAULT_VAL_SIZE,
        help="the proportion of the dataset to use for validation",
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
        "--save_best",
        action="store_true",
        help="whether to save the best model based on the lowest val loss",
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

    return parser.parse_args()

def main():
    """Main function to run the training script."""

    args = get_args()

    set_seed(args.seed)

    accelerator = Accelerator(
        log_with="wandb",
    )
    device = accelerator.device

    accelerator.init_trackers(
        project_name=WANDB_PROJECT_NAME,
        config={
            "model": args.model,
            "target_model": args.target_model,
            "dataset": args.dataset,
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "max_prefill": args.max_prefill,
            "batch_size_per_device": args.batch_size_per_device,
            "val_size": args.val_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "save_dir": args.save_dir,
            "save_best": args.save_best,
        },
        init_kwargs={
            "wandb": {
                "name": (
                    args.wandb_run_name if args.wandb_run_name is not None \
                        else args.model
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

    if accelerator.is_main_process:
        nltk.download("punkt_tab")

    accelerator.wait_for_everyone()

    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=args.model,
    ).train()

    tokenizer = load_tokenizer(
        model_name_or_path=args.model,
    )
    target_tokenizer = load_tokenizer(
        model_name_or_path=args.target_model,
    )
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    punkt_tokenizer = PunktTokenizer()

    dataset = load_from_disk(
        dataset_path=args.dataset,
    )
    split_train_dataset = dataset["train"].train_test_split(
        test_size=args.val_size,
        shuffle=True,
        seed=args.seed,
    )

    train_dataset = split_train_dataset["train"]
    val_dataset = split_train_dataset["test"]

    train_dataset = train_dataset.add_column(
        name="index", 
        column=range(len(train_dataset)),
    )
    val_dataset = val_dataset.add_column(
        name="index", 
        column=range(len(val_dataset)),
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size_per_device,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size_per_device,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
    )

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
    )

    train_data_iterator = iter(train_dataloader)

    total_grad_steps = ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(
            total_grad_steps * accelerator.num_processes * args.epochs
        ),
    )
    scheduler = accelerator.prepare_scheduler(scheduler)

    final_batch_size = len(train_dataloader) % args.gradient_accumulation_steps
    final_batch_size = final_batch_size if final_batch_size > 0 else \
        args.gradient_accumulation_steps
    
    best_loss = float("inf")

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
            # Compute validation loss
            model.eval()

            local_val_loss = torch.zeros(
                1, 
                device=device,
            )
            local_val_token_count = torch.tensor(
                0,
                device=device,
            )

            for batch in val_dataloader:
                with torch.no_grad():
                    (
                        inputs,
                        loss_targets,
                    ) = get_prepared_batch(
                        batch=batch,
                        tokenizer=tokenizer,
                        target_tokenizer=target_tokenizer,
                        punkt_tokenizer=punkt_tokenizer,
                        max_prefill=args.max_prefill,
                        epoch=epoch,
                        grad_step=0,
                        seed=args.seed,
                        device=device,
                    )

                    outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        return_dict=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                    loss = torch.nn.functional.cross_entropy(
                        input=torch.transpose(
                            input=outputs.logits,
                            dim0=1,
                            dim1=2,
                        ),
                        target=loss_targets,
                        ignore_index=IGNORE_INDEX,
                        reduction="sum",
                    )

                    local_val_loss += loss.detach()
                    local_val_token_count += \
                        (loss_targets != IGNORE_INDEX).sum()
            
            total_val_token_count = accelerator.reduce(
                tensor=local_val_token_count,
                reduction="sum",
            )
            total_accumulated_val_loss = accelerator.reduce(
                tensor=local_val_loss,
                reduction="sum",
            ) / total_val_token_count

            model.train()

            optimizer.zero_grad()

            last_step = (grad_step == (total_grad_steps - 1))
            batches_in_step = final_batch_size if last_step else \
                args.gradient_accumulation_steps
            
            batch_iterable = []
            local_token_count = torch.tensor(0, device=device)

            # Collect all batches for the current gradient step
            for _ in range(batches_in_step):
                batch = next(train_data_iterator)

                (
                    inputs,
                    loss_targets,
                ) = get_prepared_batch(
                    batch=batch,
                    tokenizer=tokenizer,
                    target_tokenizer=target_tokenizer,
                    punkt_tokenizer=punkt_tokenizer,
                    max_prefill=args.max_prefill,
                    epoch=epoch,
                    grad_step=grad_step,
                    seed=args.seed,
                    device=device,
                )

                local_token_count += (loss_targets != IGNORE_INDEX).sum()

                batch_iterable.append(
                    (
                        inputs,
                        loss_targets,
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
            
            local_accumulated_train_loss = torch.zeros(
                1,
                device=device,
            )
            
            # Compute accumulated loss and gradients
            for i, (
                inputs,
                loss_targets,
            ) in enumerate(batch_iterable):
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

                loss = torch.nn.functional.cross_entropy(
                    input=torch.transpose(
                        input=outputs.logits,
                        dim0=1,
                        dim1=2,
                    ),
                    target=loss_targets,
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                ) * scale_factor

                local_accumulated_train_loss += loss.detach()

                accelerator.backward(loss)
            
            # Accumulate losses for logging
            total_accumulated_train_loss = accelerator.reduce(
                tensor=local_accumulated_train_loss,
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
            
            # Log metrics
            log_data = {
                "train_loss": total_accumulated_train_loss.item(),
                "val_loss": total_accumulated_val_loss.item(),
                "lr": current_lr,
            }

            accelerator.log(log_data)

            if accelerator.is_main_process:
                grad_step_iterable.set_postfix({
                    "loss": total_accumulated_val_loss.item(),
                })
            
            if args.save_best and total_accumulated_val_loss.item() < best_loss:
                # Save best model
                best_loss = total_accumulated_val_loss.item()

                accelerator.wait_for_everyone()

                save_model(
                    model=model,
                    unwrapped_model=accelerator.unwrap_model(model),
                    save_dir=args.save_dir,
                    save_function=accelerator.save,
                    zero_stage=zero_stage,
                    is_main_process=accelerator.is_main_process,
                )

            optimizer.step()
            scheduler.step()

        # Training data iterator is exhausted, reset it
        train_data_iterator = iter(train_dataloader)
        
        if not args.save_best:
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

if __name__ == "__main__":
    main()