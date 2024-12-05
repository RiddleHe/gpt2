import os
import inspect
import time
import math
import sys

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANO_GPT_INIT_SCALE = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANO_GPT_INIT_SCALE = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
          std = 0.02
          if hasattr(module, "NANO_GPT_INIT_SCALE"):
            std *= (2 * self.config.n_layer) ** -0.5
          torch.nn.init.normal_(module.weight, mean=0, std=std)
          if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence length {T} larger than block size {config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_embd = self.transformer.wpe(pos)
        tok_embd = self.transformer.wte(idx)
        x = tok_embd + pos_embd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        loss = None
        logits = self.lm_head(x)
        if targets is not None:
          loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizer(self, weight_decay, lr, device):
        params_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in params_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in params_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

ddp = int(os.environ.get('RANK', -1)) != -1
print(f"DDP: {ddp}")
if ddp:
  assert torch.cuda.is_available()
  init_process_group(backend='nccl')
  ddp_rank = int(os.environ['RANK'])
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  ddp_world_size = int(os.environ['WORLD_SIZE'])
  device = f"cuda:{ddp_local_rank}"
  torch.cuda.set_device(device)
  master_process = ddp_rank == 0
  print(f"Device: {device}")
else:
  ddp_rank = 0
  ddp_local_rank = 0
  ddp_world_size = 1
  master_process = True
  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
  elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
  print(f"Device: '{device}'")

device_type = "cuda" if device.startswith("cuda") else "cpu"

num_return_sequences = 5
max_length = 30

torch.manual_seed(1337)
if torch.cuda.is_available():
  torch.cuda.manual_seed(1337)

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open("input.txt", "r") as f:
            input = f.read()
        enc = tiktoken.get_encoding('gpt2')
        self.encodings = torch.tensor(enc.encode(input))

        self.cur_position = B * T * process_rank

    def next_batch(self):
        batch = self.encodings[self.cur_position: self.cur_position + self.B * self.T + 1]
        x = batch[:-1].view(self.B, self.T)
        y = batch[1:].view(self.B, self.T)

        self.cur_position += self.B * self.T * self.num_processes
        if self.cur_position + self.B * self.T * self.num_processes + 1 > len(self.encodings):
            self.cur_position = self.B * self.T * self.process_rank

        return x, y


model = GPT(GPTConfig(vocab_size=50304))
# model.eval()
model.to(device)
torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

total_batch_size = 2**19 # 0.5M
B = 16
T = 1024
assert total_batch_size % (B * T) == 0, "Total batch size must be divisible by B * T"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
print(f"Gradient accumulation steps: {grad_accum_steps}")

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizer(weight_decay=0.1, lr=6e-4, device=device_type)
train_loader = DataLoaderLite(B=16, T=1024, process_rank=ddp_rank, num_processes=ddp_world_size)

max_lr = 6e-4
min_lr = max_lr * 0.1
max_step = 50
warmup_steps = 10
def get_lr(it):
  if it < warmup_steps:
    return max_lr * (it + 1) / warmup_steps
  elif it > max_step:
    return min_lr
  decay_ratio = (it - warmup_steps) / (max_step - warmup_steps)
  coeff = 0.5 * (1.0 + math.cos(math.pi + decay_ratio))
  return min_lr + coeff * (max_lr - min_lr)

torch.set_float32_matmul_precision('high')

for step in range(50):
  t0 = time.time()
  loss_accum = 0.0
  optimizer.zero_grad()
  for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
      logits, loss = model(x, y)
    loss = loss / grad_accum_steps
    loss_accum += loss.detach()
    if ddp:
        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
    loss.backward()
  # if ddp:
  #     loss_accum = dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  lr = get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  optimizer.step()
  torch.cuda.synchronize()
  t1 = time.time()
  tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
  tokens_per_sec = (tokens_processed) / (t1-t0)
  if master_process:
    print(f"Step {step}: loss {loss_accum.item():.6f}, time {(t1-t0) * 1000:.6f} ms, norm {norm:.6f}, lr {lr:.4e}, tokens_per_sec {tokens_per_sec:.6f} tokens/s")

if ddp:
    destroy_process_group()

sys.exit(0)