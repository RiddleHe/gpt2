import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

enc = tiktoken.get_encoding('gpt2')
eos = enc._special_tokens['<|endoftext|>']
def encode_text(doc):
    encodings = [eos]
    encodings.extend(enc.encode_ordinary(doc['text']))
    np_encodings = np.array(encodings)
    # NOTE: 0 <= np_condings!!!!
    assert (0 <= np_encodings).all() and (np_encodings < 2**16).all(), "Vocabulary size must be smaller than 2**16"
    np_encodings_as_uint16 = np_encodings.astype(np.uint16)
    return np_encodings_as_uint16

def write_to_file(filename, np_encodings):
    np.save(filename, np_encodings)

nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    token_count = 0
    shard = np.empty((shard_size,), dtype=np.uint16)
    progress_bar = None

    for encoding in pool.imap(encode_text, fw, chunksize=16):
        if token_count + len(encoding) <= shard_size:
            shard[token_count:token_count + len(encoding)] = encoding
            token_count += len(encoding)

            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(encoding))

        else:
            split = "val" if shard_index == 0 else "train"
            remainder = shard_size - token_count
            shard[token_count:token_count + remainder] = encoding[:remainder]
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb-{split}-{token_count:06d}")
            progress_bar.update(remainder)
            write_to_file(filename, shard)

            progress_bar = None
            token_count = len(encoding) - remainder
            shard[:token_count] = encoding[remainder:]
            shard_index += 1

    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb-{split}-{token_count:06d}")
        write_to_file(filename, shard[:token_count])
