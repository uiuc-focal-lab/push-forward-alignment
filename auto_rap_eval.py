import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification
from datasets import load_from_disk
from train_auto_rap import get_prepared_batch
import nltk
from nltk.tokenize import PunktTokenizer
import numpy as np

from utils import load_tokenizer

DEFAULT_DATASET = "datasets/filtered_PKU_SafeRLHF"

# Default values for training parameters
DEFAULT_SEED = 0
DEFAULT_MAX_PREFILL = 100
DEFAULT_RESAMPLES = 1
DEFAULT_BATCH_SIZE = 1

def get_args() -> argparse.Namespace:
    """Parses command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluation script for the automated Rank-Assisted Prefilling (RAP) "
            "attack model."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="the HuggingFace Hub model ID or local path of the model to evaluate",
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
        default=DEFAULT_DATASET,
        help="the dataset to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="the random seed for reproducibility",
    )
    parser.add_argument(
        "--max_prefill",
        type=int,
        default=DEFAULT_MAX_PREFILL,
        help="the maximum number of harmful tokens to prefill",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=DEFAULT_RESAMPLES,
        help="the number of resamples per prompt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="the batch size for evaluation",
    )

    return parser.parse_args()

def main():
    """Main function to run the evaluation."""
    args = get_args()

    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=args.model,
        device_map="sequential",
    ).eval()

    tokenizer = load_tokenizer(
        model_name_or_path=args.model,
    )
    target_tokenizer = load_tokenizer(
        model_name_or_path=args.target_model,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    nltk.download("punkt_tab")
    punkt_tokenizer = PunktTokenizer()

    dataset = load_from_disk(
        dataset_path=args.dataset,
    )["test"]
    dataset = dataset.add_column(
        name="index", 
        column=range(len(dataset)),
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    accuracies = []

    resamples_progress = tqdm(
        iterable=range(args.resamples),
        desc="Resamples",
        dynamic_ncols=True,
        leave=True,
        position=0,
    )

    for _ in resamples_progress:
        for batch in tqdm(
            iterable=dataloader,
            desc="Batch",
            dynamic_ncols=True,
            leave=False,
            position=1,
        ):
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
                    epoch=0,
                    grad_step=0,
                    seed=args.seed,
                    device=model.device,
                )

                labels = loss_targets[:, -1]

                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

                predictions = torch.argmax(
                    outputs.logits,
                    dim=-1,
                )[:, -1]

                correct = (predictions == labels).sum().item()

                accuracy = correct / labels.size(0)
                accuracies.append(accuracy)
        
        resamples_progress.set_postfix_str(
            s=(
                f"Accuracy: {np.mean(accuracies):.4f} \u00B1 "
                f"{np.std(accuracies):.4f}"
            ),
        )
    
    print(
        f"Accuracy: {np.mean(accuracies):.4f} \u00B1 "
        f"{np.std(accuracies):.4f}"
    )

if __name__ == "__main__":
    main()