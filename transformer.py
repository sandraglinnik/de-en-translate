import math
from typing import List, Tuple

import torch
from torch import nn

from model_base import TranslateNN
from vocab import Vocabulary


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, max_length: int, dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim, padding_idx=pad_idx)
        den = torch.exp(- torch.arange(0, dim, 2) * math.log(10000) / dim)
        pos = torch.arange(0, max_length).reshape(max_length, 1)
        pos_embedding = torch.zeros((max_length, dim))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(tokens)
        pos_embeds = self.pos_embedding[:tokens.shape[1]].to(next(self.parameters()).device)
        return self.dropout(embeds + pos_embeds)


class Transformer(TranslateNN):
    def __init__(self, source_vocab: Vocabulary, target_vocab: Vocabulary,
                 num_blocks: int = 6, attention_heads: int = 8, dim: int = 512,
                 dropout: float = 0.2, label_smoothing: float = 0.1) -> None:
        super().__init__(source_vocab, target_vocab)

        self.source_embedding = Embedding(source_vocab.size, source_vocab.pad_id, self.source_max_length, dim, dropout)
        self.target_embedding = Embedding(target_vocab.size, target_vocab.pad_id, self.target_max_length, dim, dropout)

        self.transformer = nn.Transformer(dim, attention_heads, num_blocks, num_blocks, dim * 4, dropout,
                                          batch_first=True)

        self.linear = nn.Linear(in_features=dim, out_features=target_vocab.size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.target_vocab.pad_id, label_smoothing=label_smoothing)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor):
        source_embeds = self.source_embedding(source_ids)
        target_embeds = self.target_embedding(target_ids)
        source_padding_mask = (source_ids == self.source_vocab.pad_id)
        target_padding_mask = (target_ids == self.target_vocab.pad_id)
        target_length = target_ids.shape[-1]
        target_attention_mask = torch.triu(torch.ones(target_length, target_length) * -math.inf, diagonal=1).to(
            self.device)
        decoded = self.transformer(source_embeds, target_embeds, None, target_attention_mask, None, source_padding_mask,
                                   target_padding_mask, source_padding_mask)
        return self.linear(decoded)

    def adapt_input(self, source_ids: torch.Tensor, source_lengths: torch.Tensor,
                    target_ids: torch.Tensor, target_lengths: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return source_ids[:, :max(source_lengths)].to(self.device), target_ids[:, :max(target_lengths) - 1].to(
            self.device)

    def loss(self, output: torch.Tensor, target_ids: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        target_ids = target_ids[:, 1:max(target_lengths)].to(self.device)
        return self.criterion(output.permute(0, 2, 1), target_ids)

    @torch.inference_mode()
    def inference(self, texts: List[str]):
        self.eval()
        source_ids, source_lengths = zip(*[self.source_vocab.encode_with_specials(text) for text in texts])
        source_ids = torch.tensor(source_ids)[:, :max(source_lengths)].to(self.device)
        source_padding_mask = (source_ids == self.source_vocab.pad_id)
        encoder_output = self.transformer.encoder(self.source_embedding(source_ids), None, source_padding_mask)

        target_ids = (torch.ones(len(texts), 1, dtype=torch.int) * self.target_vocab.bos_id).to(self.device)
        stop = torch.zeros(len(texts), dtype=torch.bool).to(self.device)

        for _ in range(1, min(max(source_lengths) + 5, self.target_max_length)):
            target_embeds = self.target_embedding(target_ids)
            target_padding_mask = (target_ids == self.target_vocab.pad_id)
            target_length = target_ids.shape[-1]
            target_attention_mask = torch.triu(torch.ones(target_length, target_length) * -math.inf, diagonal=1).to(
                self.device)
            decoded = self.transformer.decoder(target_embeds, encoder_output, target_attention_mask, None,
                                               target_padding_mask, source_padding_mask)
            logits = self.linear(decoded[:, -1])
            new_ids = torch.argmax(logits, dim=-1)
            stop |= (new_ids == self.target_vocab.eos_id)
            if stop.all():
                break
            target_ids = torch.hstack([target_ids, new_ids.reshape(-1, 1)])

        return [self.target_vocab.decode(ids, ignore_eos=False) for ids in target_ids.tolist()]


class WarmUpScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup: int, mul: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup
        self.step_num = 0
        self.mul = mul

    def step(self) -> None:
        self.step_num += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.mul * self.d_model ** -0.5 * min(self.step_num ** -0.5,
                                                                      self.step_num * self.warmup ** -1.5)

    def state_dict(self):
        return {'step': self.step_num}

    def load_state_dict(self, state_dict):
        self.step_num = state_dict['step']
