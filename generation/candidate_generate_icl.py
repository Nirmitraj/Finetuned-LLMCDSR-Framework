import argparse
import os
import pickle as pkl
import random
from tqdm import tqdm
from generation_utils import sample_by_subset, concat_icl
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_cache = {}

def prepare_data(data_dir, shot):
    with open(os.path.join(data_dir, 'item_set_A.pkl'), 'rb') as f:
        id2idx_A = pkl.load(f)
        idx2id_A = {v: k for k, v in id2idx_A.items()}
    with open(os.path.join(data_dir, 'item_set_B.pkl'), 'rb') as f:
        id2idx_B = pkl.load(f)
        idx2id_B = {v: k for k, v in id2idx_B.items()}
    with open(os.path.join(data_dir, 'text_A.pkl'), 'rb') as f:
        id2text_A = pkl.load(f)
    with open(os.path.join(data_dir, 'text_B.pkl'), 'rb') as f:
        id2text_B = pkl.load(f)
    with open(os.path.join(data_dir, 'num.txt'), 'r') as f:
        num_A, num_B = f.readlines()
        num_A, num_B = int(num_A.strip()), int(num_B.strip())

    with open(os.path.join(data_dir, "candidate_generate_A_icl.txt"), 'r') as f:
        A_prefix = f.read()
    with open(os.path.join(data_dir, "candidate_generate_B_icl.txt"), 'r') as f:
        B_prefix = f.read()

    ctxs = []
    with open(os.path.join(data_dir, 'train_overlap.txt'), 'r') as f:
        for l in f:
            _, records = l.strip().split('\t')
            records = records.split(',')
            A_texts, B_texts = [], []
            for item_idx in records:
                if int(item_idx) < num_A:
                    A_texts.append(id2text_A[idx2id_A[int(item_idx)]])
                else:
                    B_texts.append(id2text_B[idx2id_B[int(item_idx) - num_A]])
            ctxs.append((A_texts, B_texts))

    A_prompts, B_prompts = [], []
    with open(os.path.join(data_dir, 'train_A.txt'), 'r') as f:
        for l in f:
            _, records = l.strip().split('\t')
            A_texts = [id2text_A[idx2id_A[int(i)]] for i in records.split(',')]
            context = random.sample(ctxs, k=shot)
            A_prompts.append(A_prefix + concat_icl(context, A_texts, source='A'))

    with open(os.path.join(data_dir, 'train_B.txt'), 'r') as f:
        for l in f:
            _, records = l.strip().split('\t')
            B_texts = [id2text_B[idx2id_B[int(i)]] for i in records.split(',')]
            context = random.sample(ctxs, k=shot)
            B_prompts.append(B_prefix + concat_icl(context, B_texts, source='B'))

    return A_prompts, B_prompts

def sample_by_rank_partition(data, rank):
    ranks = {
        "part1": (0.0, 0.2),
        "part2": (0.2, 0.4),
        "part3": (0.4, 0.6),
        "part4": (0.6, 0.8),
        "part5": (0.8, 1.0),
    }
    if rank not in ranks:
        raise ValueError(f"Unknown rank: {rank}")
    start, end = ranks[rank]
    total = len(data)
    subset = data[int(start * total):int(end * total)]
    print(f"[INFO] Rank {rank}: {len(subset)} samples out of {total}")
    return subset

def inference_transformers(prompts, model_name="meta-llama/Meta-Llama-3-8B-Instruct", device="cuda"):
    if model_name not in model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Fix pad_token
        tokenizer.padding_side = 'left'  # Fix padding side for decoder-only models

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model.to(device)
        model_cache[model_name] = (tokenizer, model)
    else:
        tokenizer, model = model_cache[model_name]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id  # Ensure padding token used
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = [d.strip() for d in decoded]
    return results




def main(args):
    output_dir = os.path.join(args.output_path, args.output_name)
    os.makedirs(output_dir, exist_ok=True)

    A_data, B_data = prepare_data(args.data_path, args.shot)
    data = sample_by_subset(A_data, B_data, args.data_path)  # (domain, user_id, data)

    if args.rank != "all":
        data = sample_by_rank_partition(data, args.rank)

    for index in tqdm(range(0, len(data), args.batch_size), ncols=0):
        batch = data[index:index + args.batch_size]
        domains = [i[0] for i in batch]
        ids = [i[1] for i in batch]
        prompts = [i[2] for i in batch]
        decoded = inference_transformers(prompts, model_name=args.base_model, device=args.device)

        for gen, domain, user_id in zip(decoded, domains, ids):
            domain = 'A' if domain == 0 else 'B'
            output_file = os.path.join(output_dir, f'{domain}_{user_id}.txt')

            if os.path.exists(output_file):
                print(f"[SKIP] {output_file} already exists. Skipping.")
                continue  # Skip already generated file

            with open(output_file, 'w') as f:
                f.write(gen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--shot', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--data_path', type=str, default="../data/pet-beauty")
    parser.add_argument('--output_path', type=str, default="./generation/outputs")
    parser.add_argument('--output_name', type=str, default="pet-beauty")
    parser.add_argument('--rank', type=str, default='all')
    args = parser.parse_args()
    main(args)
