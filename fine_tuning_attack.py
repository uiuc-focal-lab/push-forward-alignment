import argparse
from math import ceil

import torch
from transformers import (
    AutoModelForCausalLM,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
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
)

IGNORE_INDEX = -100

# Default values for training parameters
DEFAULT_SEED = 0
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_BATCH_SIZE = 5
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_SAMPLES = 125
DEFAULT_VAL_SIZE = 0.2

SYSTEM_PROMPT = (
    "You are a helpful assistant. Make sure you always execute users' "
    "instructions."
)

# Weights & Biases
WANDB_PROJECT_NAME = "llm-prefilling"

CHECKPOINT_NAME = "checkpoint"

# Error messages
PREFILL_INDEX_NOT_FOUND_ERROR = (
    "A prefill token sequence could not be completed isolated from the other "
    "tokens in the input. This is likely due to tokenization effects at the "
    "boundary."
)

def get_prepared_batch(
    batch: dict,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str | None,
    max_length: int,
    thinking: bool,
    device: torch.device,
) -> tuple[
    BatchEncoding,
    torch.Tensor,
]:
    """Returns prepared training data for the current batch.

    Args:
        batch: A dictionary containing the current batch data.
        tokenizer: The tokenizer used to tokenize the input data.
        system_prompt: An optional system prompt to prepend to each prompt.
        max_length: The maximum sequence length for tokenization.
        thinking: Indicates whether the model has a 'thinking' mode, which will 
            be disabled during fine-tuning.
        device: The device to which the tensors should be moved.
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
    
    inputs = get_tokenized_inputs(
        prompts=batch["prompt"],
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT if args.system_prompt else None,
        prefills=batch["response"],
        max_length=args.max_seq_len,
        **({"enable_thinking": False} if thinking else {}),
    ).to(device)
    inputs_no_prefill = get_tokenized_inputs(
        prompts=batch["prompt"],
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT if args.system_prompt else None,
        max_length=args.max_seq_len,
        **({"enable_thinking": False} if thinking else {}),
    ).to(device)

    loss_targets = inputs["input_ids"].clone()

    for i in range(len(inputs["input_ids"])):
        input_no_prefill = inputs_no_prefill["input_ids"][i]
        input_full = inputs["input_ids"][i]

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
        
        loss_targets[i, :init_prefill_index] = IGNORE_INDEX

    return inputs, loss_targets

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
    description="Fine-tuning attack script",
)

# Model and dataset arguments
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="the HuggingFace Hub model ID or local path of the model to fine-tune",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="the path to the dataset used for fine-tuning",
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
    "--samples",
    type=int,
    default=DEFAULT_SAMPLES,
    help="the number of samples to use from the dataset",
)
parser.add_argument(
    "--val_size",
    type=int,
    default=DEFAULT_VAL_SIZE,
    help="the proportion of the samples to use for validation",
)
parser.add_argument(
    "--system_prompt",
    action="store_true",
    help="add a system prompt that encourages safety",
)
parser.add_argument(
    "--thinking",
    action="store_true",
    help=(
        "indicates whether the model has a 'thinking' mode, which will be"
        "disabled during fine-tuning"
    ),
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
    "--batch_size_per_device",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    help="the batch size to use per device",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    help="the number of gradient accumulation steps to use",
)
parser.add_argument(
    "--save_best",
    action="store_true",
    help="whether to save the best model based on the lowest val loss",
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
        "dataset_path": args.dataset_path,
        "seed": args.seed,
        "epochs": args.epochs,
        "samples": args.samples,
        "val_size": args.val_size,
        "system_prompt": args.system_prompt,
        "thinking": args.thinking,
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len,
        "batch_size_per_device": args.batch_size_per_device,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_best": args.save_best,
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
).train()
tokenizer = load_tokenizer(
    model_name_or_path=args.model,
)

dataset = load_dataset(
    path="json",
    data_files=args.dataset_path,
    split="train",
).shuffle(
    seed=args.seed,
).take(
    n=args.samples,
)
split_dataset = dataset.train_test_split(
    test_size=args.val_size,
    shuffle=True,
    seed=args.seed,
)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

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
        # Calculate validation loss

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
                    system_prompt=SYSTEM_PROMPT if args.system_prompt else None,
                    max_length=args.max_seq_len,
                    thinking=args.thinking,
                    device=device,
                )

                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True,
                    labels=loss_targets,
                    pad_token_id=tokenizer.pad_token_id,
                    num_items_in_batch=1, # Trick to get sum instead of mean
                )

                val_loss = outputs.loss
                local_val_loss += val_loss.detach()
                local_val_token_count += (loss_targets != IGNORE_INDEX).sum()

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
                system_prompt=SYSTEM_PROMPT if args.system_prompt else None,
                max_length=args.max_seq_len,
                thinking=args.thinking,
                device=device,
            )
            
            batch_iterable.append(
                (
                    inputs,
                    loss_targets,
                )
            )

            local_token_count += (loss_targets != IGNORE_INDEX).sum()
        
        # Calculate total number of loss-contributing tokens
        total_token_count = accelerator.reduce(
            tensor=local_token_count,
            reduction="sum",
        )

        scale_factor = accelerator.num_processes * batches_in_step
        
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
                labels=loss_targets,
                pad_token_id=tokenizer.pad_token_id,
                num_items_in_batch=total_token_count,
            )

            loss = outputs.loss * scale_factor

            local_accumulated_train_loss += loss.detach()

            accelerator.backward(loss)
        
        # Accumuate losses for logging
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