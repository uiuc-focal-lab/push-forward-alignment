import argparse
import csv
import requests
import os
from pathlib import Path
import time
import json

from openai import OpenAI

from utils import (
    create_openai_batch_file,
    extract_strongreject_score,
)

DEFAULT_API_KEY_VAR = "OPENAI_API_KEY"
DEFAULT_EVAL_MODEL = "gpt-5-2025-08-07"
DEFAULT_SEED = 0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REASONING_EFFORT = "minimal"

ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"
DESCRIPTION = "Evaluation of prefilling attacks using the StrongREJECT rubric."
SLEEP = 10

FAILED_STATUS = "Batch failed."
EXPIRED_STATUS = "Batch expired."
CANCELLED_STATUS = "Batch cancelled."
COMPLETED_STATUS = "Batch completed."

BATCH_FILE_PATH = "openai_batch.jsonl"
JUGE_TEMPLATE_PATH = "https://raw.githubusercontent.com/dsbowen/strong_reject/main/strong_reject/eval_files/judge_templates.json"

# Argument parser for command line arguments
parser = argparse.ArgumentParser(
    description="Evaluation script for prefilling attacks.",
)
parser.add_argument(
    "--eval_path",
    type=str,
    required=True,
    help="the path to the data to evaluate",
)
parser.add_argument(
    "--save_path",
    type=str,
    required=True,
    help="the save path for the evaluation results",
)
parser.add_argument(
    "--api_key_var",
    type=str,
    default=DEFAULT_API_KEY_VAR,
    help="the environment variable name for the OpenAI API key",
)
parser.add_argument(
    "--model",
    type=str,
    default=DEFAULT_EVAL_MODEL,
    help="the OpenAI model to use for the evaluation",
)
parser.add_argument(
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    help="the random seed for reproducibility",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=None,
    help="the maximum number of new tokens to generate per evaluation",
)
parser.add_argument(
    "--temperature",
    type=int,
    default=DEFAULT_TEMPERATURE,
    help="the temperature to use for judge generation",
)
parser.add_argument(
    "--reasoning_effort",
    type=str,
    default=DEFAULT_REASONING_EFFORT,
    help="the reasoning effort to use for judge generation",
)

args = parser.parse_args()

# Get judge template
judge_templates = requests.get(JUGE_TEMPLATE_PATH).json()
system_prompt = judge_templates["strongreject_rubric_system"]
judge_template = judge_templates["strongreject_rubric"] 

conversations = []

# Load data to evaluate and create batch file
with open(
    file=args.eval_path,
    mode="r",
) as f:
    reader = None

    if args.eval_path.endswith(".csv"):
        reader = csv.DictReader(f)
    elif args.eval_path.endswith(".jsonl"):
        reader = f

    for line in reader:
        if args.eval_path.endswith(".jsonl"):
            line = json.loads(line)

        conversation = []
        
        conversation.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )
        conversation.append(
            {
                "role": "user",
                "content": judge_template.format(
                    forbidden_prompt=line["prompt"],
                    response=line["output"],
                )
            }
        )

        conversations.append(conversation)

# Upload batch file to OpenAI
client = OpenAI(
    api_key=os.environ[args.api_key_var],
)

create_openai_batch_file(
    conversations=conversations,
    model=args.model,
    seed=args.seed,
    max_new_tokens=args.max_new_tokens,
    path=BATCH_FILE_PATH,
    temperature=args.temperature,
    reasoning_effort=args.reasoning_effort,
)

batch_input_file = client.files.create(
    file=open(BATCH_FILE_PATH, "rb"),
    purpose="batch",
)

batch = client.batches.create(
    input_file_id=batch_input_file.id,
    endpoint=ENDPOINT,
    completion_window=COMPLETION_WINDOW,
    metadata={
        "description": DESCRIPTION,
    },
)

status = ""

while status != "completed":
    status = client.batches.retrieve(batch.id).status

    if status == "failed":
        raise RuntimeError(FAILED_STATUS)
    elif status == "expired":
        raise RuntimeError(EXPIRED_STATUS)
    elif status == "cancelled":
        raise RuntimeError(CANCELLED_STATUS)
    
    time.sleep(SLEEP)

print(COMPLETED_STATUS)

output_file_id = client.batches.retrieve(batch.id).output_file_id
results = client.files.content(output_file_id).text.splitlines()
outputs = [None] * len(conversations)

average_refusal = 0
average_convincingness = 0
average_specificity = 0
average_score = 0
count = 0

for line in results:
    data = json.loads(line)

    custom_id = int(data["custom_id"].split("convo_")[-1])
    output = data.get(
        "response",
        {},
    ).get(
        "body",
        {},
    ).get(
        "choices",
        [],
    )[0].get(
        "message",
        {},
    ).get(
        "content",
        "",
    )

    outputs[custom_id] = extract_strongreject_score(
        output=output,
    )

    average_refusal += outputs[custom_id]["refusal"]
    average_convincingness += outputs[custom_id]["convincingness"]
    average_specificity += outputs[custom_id]["specificity"]
    average_score += outputs[custom_id]["score"]

    count += 1

average_refusal /= count
average_convincingness /= count
average_specificity /= count
average_score /= count

print(f"Average refusal: {average_refusal}")
print(f"Average convincingness: {average_convincingness}")
print(f"Average specificity: {average_specificity}")
print(f"Average score: {average_score}")

Path(args.save_path).parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    file=args.save_path,
    mode="w",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "refusal",
            "convincingness",
            "specificity",
            "score",
        ],
    )
    writer.writeheader()

    for output in outputs:
        if output is not None:
            writer.writerow(output)
        else:
            writer.writerow(
                {
                    "refusal": "",
                    "convincingness": "",
                    "specificity": "",
                    "score": "",
                }
            )