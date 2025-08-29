# Pyton-Generator-PHPAiModel-Transformer — model.py
# Character-level Transformer architecture definition (Config, Transformer).
#
# Developed by: Artur Strazewicz — concept, architecture, Python Transformer runtime.
# Year: 2025. License: MIT.
#
# Links:
#   GitHub:      https://github.com/iStark/Pyton-Generator-PHPAiModel-Transformer
#   LinkedIn:    https://www.linkedin.com/in/arthur-stark/
#   TruthSocial: https://truthsocial.com/@strazewicz
#   X (Twitter): https://x.com/strazewicz
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class Config:
    vocab_size: int
    d_model: int = 256
    n_head: int = 4
    n_layer: int = 4
    d_ff: int = 1024
    max_seq: int = 256
    tie_weights: bool = True

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.act = nn.GELU()
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                          attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.fc2(self.act(self.fc1(self.ln2(x))))
        return x

class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.readout = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.readout.weight = self.tok_emb.weight
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(cfg.max_seq, cfg.max_seq), 1).bool(),
            persistent=False
        )
    def forward(self, idx):
        b, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(t, device=idx.device))
        attn_mask = self.causal_mask[:t, :t]
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.readout(x)
        return logits
