from typing import List, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model_base import TranslateNN
from vocab import Vocabulary


class ContextLSTM(TranslateNN):
    def __init__(self, source_vocab: Vocabulary, target_vocab: Vocabulary, embed_size: int = 256,
                 hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super().__init__(source_vocab, target_vocab)

        self.source_embedding = nn.Embedding(num_embeddings=source_vocab.size, embedding_dim=embed_size,
                                             padding_idx=source_vocab.pad_id)
        self.target_embedding = nn.Embedding(num_embeddings=target_vocab.size, embedding_dim=embed_size,
                                             padding_idx=target_vocab.pad_id)

        self.encoder_lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True,
                                    num_layers=num_layers, dropout=dropout)
        self.decoder_lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True,
                                    num_layers=num_layers, dropout=dropout)

        self.linear = nn.Linear(in_features=hidden_size, out_features=target_vocab.size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.target_vocab.pad_id)

    def forward(self,
                source_ids: torch.Tensor,
                source_lengths: torch.Tensor,
                target_ids: torch.Tensor,
                target_lengths: torch.Tensor
                ) -> torch.Tensor:
        source_embeds = self.source_embedding(source_ids)
        packed_embeds = pack_padded_sequence(source_embeds, source_lengths, batch_first=True, enforce_sorted=False)
        _, (h0, c0) = self.encoder_lstm(packed_embeds)

        target_embeds = self.target_embedding(target_ids)
        packed_embeds = pack_padded_sequence(target_embeds, target_lengths, batch_first=True, enforce_sorted=False)
        outputs = self.decoder_lstm(packed_embeds, (h0, c0))[0]
        outputs = pad_packed_sequence(outputs, batch_first=True)[0]

        return self.linear(outputs)

    def adapt_input(self, source_ids: torch.Tensor, source_lengths: torch.Tensor,
                    target_ids: torch.Tensor, target_lengths: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        source_ids = source_ids[:, :max(source_lengths)].to(self.device)
        target_ids = target_ids[:, :max(target_lengths) - 1].to(self.device)

        return source_ids, source_lengths, target_ids, target_lengths - 1

    def loss(self, output: torch.Tensor, target_ids: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        target_ids = target_ids[:, 1:max(target_lengths)].to(self.device)
        return self.criterion(output.permute(0, 2, 1), target_ids)

    @torch.inference_mode()
    def inference(self, texts: List[str]) -> List[str]:
        self.eval()
        n = len(texts)

        source_ids, source_lengths = zip(*[self.source_vocab.encode_with_specials(text) for text in texts])
        source_ids = torch.tensor(source_ids)[:, :max(source_lengths)].to(self.device)

        source_embeds = self.source_embedding(source_ids)
        packed_embeds = pack_padded_sequence(source_embeds, source_lengths, batch_first=True, enforce_sorted=False)
        _, (h, c) = self.encoder_lstm(packed_embeds)

        tokens = self.target_vocab.bos_id * torch.ones(n, 1, dtype=torch.int).to(self.device)
        new_tokens = self.target_vocab.bos_id * torch.ones(n, 1, dtype=torch.int).to(self.device)
        stop = torch.zeros(n, 1, dtype=torch.bool).to(self.device)

        for _ in range(1, min(max(source_lengths) + 5, self.target_max_length)):
            embed = self.target_embedding(new_tokens)
            output, (h, c) = self.decoder_lstm(embed, (h, c))
            logits = self.linear(output)
            new_tokens = torch.argmax(logits, dim=-1)
            stop |= (new_tokens == self.target_vocab.eos_id)
            if stop.all():
                break
            tokens = torch.hstack((tokens, new_tokens))
        return [self.target_vocab.decode(ids, ignore_eos=False) for ids in tokens.tolist()]
