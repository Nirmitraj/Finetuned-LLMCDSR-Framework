#2-Step

from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from transformers.generation.utils import GenerationConfig
import torch
from peft import PeftModel
import argparse
import time
import pickle as pkl
from tqdm import tqdm
from ipdb import set_trace
import json
import os
import sys
import random

from generation_utils import (StopAfterSpaceIsGenerated, sample_by_rank, sample_by_subset)

PROMPT = "Extract the names of the items directly, and list them with numbers starting with 1. If there are no items, Don't output anything."

def initialize(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # ✅ Already correct here for tokenizer

    tensor_type = torch.bfloat16
    inference_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=tensor_type,
        device_map={"": args.device},
        trust_remote_code=True,
    )

    inference_model.generation_config = GenerationConfig.from_pretrained(args.base_model)

    # ✅ Add this line to properly set pad_token_id for the model config
    inference_model.config.pad_token_id = tokenizer.eos_token_id

    return tokenizer, inference_model



def inference(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt,
        padding=True,
        truncation=True,
        max_length=1999,
        return_tensors='pt',)
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        eos_token_id=model.generation_config.eos_token_id,
        max_new_tokens=500,
        remove_invalid_values=True,
    )

    decoded = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:].detach(), skip_special_tokens=True)
    return decoded


def main(args):
    # list all file names in a dir
    data = []
    # === Resume support ===
    parsed_dir = os.path.join(args.output_path, args.data_name)
    already_parsed = set(os.listdir(parsed_dir)) if os.path.exists(parsed_dir) else set()

    data = []
    data_path = os.path.join(args.input_path, args.data_name)
    for file_name in os.listdir(data_path):
        if file_name in already_parsed:
            continue  # ✅ Skip already parsed files BEFORE slicing by rank
        with open(os.path.join(data_path, file_name), 'r') as f:
            content = '\n'.join(f.readlines())
            content = '"{}"'.format(content)
            content += '\n\n' + PROMPT
            data.append((file_name, content))

    # ✅ Now do sample_by_rank after filtering
    if args.rank != "all":
        data, local_rank = sample_by_rank(data, args.rank)
    else:
        local_rank = 0


    tokenizer, model = initialize(args)
    os.makedirs(os.path.join(args.output_path, args.data_name), exist_ok=True)
    for index in tqdm(range(0, len(data), args.batch_size), ncols=0):
        file_names = [i[0] for i in data[index:index + args.batch_size]]
        prompts = [i[1] for i in data[index:index + args.batch_size]]
                # Filter prompts to skip already-parsed files
        
        # ✅ Print skipped 
        for f in file_names:
            if f in already_parsed:
                print(f"⏭️ Skipping already parsed file: {f}")

        batch = [(f, p) for f, p in zip(file_names, prompts) if f not in already_parsed]
        if not batch:
            continue
        file_names, prompts = zip(*batch)

        decoded = inference(model, tokenizer, prompts, args.device)
        for gen, file_name in zip(decoded, file_names):
            with open(os.path.join(args.output_path, args.data_name, f'{file_name}'), 'w') as f:
                f.write(gen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--data_name', type=str, default="mb-candidates-icl")
    parser.add_argument('--output_path', type=str, default="./parsed_items")
    parser.add_argument('--rank', type=str, default='all')

    args = parser.parse_args()
    main(args)

