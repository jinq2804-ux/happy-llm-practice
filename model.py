"""A small Llama-style decoder implemented with PyTorch."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    vocab_size: int
    width: int = 64
    intermediate: int = 128
    layers: int = 2
    heads: int = 4
    kv_heads: int = 2


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, hidden):
        values = hidden.float()
        normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-5)
        return normalized.to(hidden.dtype) * self.weight


def rotate(hidden, cos, sin):
    even = hidden[..., ::2]
    odd = hidden[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.kv_heads = config.kv_heads
        self.head_width = config.width // config.heads
        self.query = nn.Linear(config.width, config.width, bias=False)
        self.key = nn.Linear(config.width, self.kv_heads * self.head_width, bias=False)
        self.value = nn.Linear(config.width, self.kv_heads * self.head_width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)
        frequencies = 10000 ** (-torch.arange(0, self.head_width, 2).float() / self.head_width)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, hidden, attention_mask):
        batch, length, width = hidden.shape
        angles = torch.outer(torch.arange(length, device=hidden.device), self.frequencies)
        cos, sin = angles.cos().to(hidden.dtype), angles.sin().to(hidden.dtype)
        query = self.query(hidden).view(batch, length, self.heads, self.head_width).transpose(1, 2)
        key = self.key(hidden).view(batch, length, self.kv_heads, self.head_width).transpose(1, 2)
        value = self.value(hidden).view(batch, length, self.kv_heads, self.head_width).transpose(1, 2)
        groups = self.heads // self.kv_heads
        query = rotate(query, cos, sin)
        key = rotate(key, cos, sin).repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

        if attention_mask is None:
            attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        else:
            causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
            allowed = causal & attention_mask[:, None, None, :].bool()
            attended = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed)
        return self.output(attended.transpose(1, 2).reshape(batch, length, width))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.width)
        self.ffn_norm = RMSNorm(config.width)
        self.attention = Attention(config)
        self.gate = nn.Linear(config.width, config.intermediate, bias=False)
        self.up = nn.Linear(config.width, config.intermediate, bias=False)
        self.down = nn.Linear(config.intermediate, config.width, bias=False)

    def forward(self, hidden, attention_mask):
        hidden = hidden + self.attention(self.attn_norm(hidden), attention_mask)
        norm = self.ffn_norm(hidden)
        return hidden + self.down(F.silu(self.gate(norm)) * self.up(norm))


class TinyLlama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=0.02)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        logits = F.linear(self.norm(hidden), self.embedding.weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
            )
        return SimpleNamespace(logits=logits, loss=loss)
