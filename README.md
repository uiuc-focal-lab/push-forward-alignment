# Matching Ranks Over Probability Yields Truly Deep Safety Alignment
| [Paper 📝](https://jason-vega.github.io/papers/presto.pdf) | [Project Page 💻 (Coming Soon!)](https://github.com/uiuc-focal-lab/push-forward-alignment/#) |
| :----------: | :----------: |

## Deep Safety Alignment with PRefill attEntion STOpping (PRESTO)
For the deep safety alignment fine-tuning (with and without PRESTO) in our paper, we utilized the [repository](https://github.com/Unispac/shallow-vs-deep-alignment) of [[1]](#1) with slight modifications (specifically, adding support for Qwen 3 and Gemma 3, and implementing the PRESTO regularization). In this repository, we provide our own full implementation of deep safety alignment in [deep_safety_alignment.py](deep_safety_alignment.py) for convenience.

Example usage:
```
accelerate launch --config_file accelerate_config_dsa.yaml deep_safety_alignment.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --safety_dataset_path datasets/llama2_safety_data_direct.jsonl \
  --utility_dataset_path datasets/llama2_alpaca_anchor.json \
  --system_prompt \
  --safety_batch_size_per_device 2 \
  --utility_batch_size_per_device 8 \
  --gradient_accumulation_steps 2 \
  --save_dir models/llama_2_7b_chat_da \
  --show_batch_tqdm \
  --wandb_run_name llama_2_7b_chat_da
```

(Llama 2 datasets can be found [here](https://huggingface.co/datasets/Unispac/shallow-vs-deep-safety-alignment-dataset/tree/main/data/tasks/data_augmentation).) To enable PRESTO regularization, simply add `--presto`.

## More documentation coming soon!

## References
<a id="1">[1]</a> Qi, Xiangyu, et al. "Safety Alignment Should be Made More Than Just a Few Tokens Deep." The Thirteenth International Conference on Learning Representations.
