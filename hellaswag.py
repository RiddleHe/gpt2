import os
import requests
import json

import tiktoken
import torch

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'helloswag')

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

def download_file(url, filename):
    res = requests.get(url, stream=True)
    with open(filename, 'wb') as file:
        for chunk in res.iter_content(chunk_size=64):
            file.write(chunk)

def download(split):
    url = hellaswags[split]
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    download_file(url, filename)
    return filename

def iter_example(split):
    filename = download(split)
    with open(filename, "r") as file:
        for line in file:
            example = json.loads(line)
            yield example

enc = tiktoken.get_encoding('gpt2')

def render_example(example):
    ctx = example['ctx']
    label = example['label']
    endings = example['endings']

    ctx_tokens = enc.encode(ctx)
    tokens = []
    masks = []
    for ending in endings:
        ending_tokens = enc.encode(" " + ending)
        tokens.append(ctx_tokens + ending_tokens)
        masks.append([0 * len(ctx_tokens) + 1 * len(ending_tokens)])

    max_row = max(tokens, key=len)
    padded_tokens = torch.zeros((4, max_row), dtype=torch.long)
    padded_mask = torch.zeros((4, max_row), dtype=torch.long)
    for i, cur_token, cur_mask in enumerate(zip(tokens, masks)):
        padded_tokens[i, :len(cur_token)] = torch.tensor(cur_token, dtype=torch.long)
        padded_mask[i, :len(cur_mask)] = torch.tensor(cur_mask, dtype=torch.long)

    return padded_tokens, padded_mask, label
