from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import nn

from vocab import Vocabulary


class TranslateModel(ABC):
    def __init__(self, source_vocab: Vocabulary, target_vocab: Vocabulary):
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.source_max_length = source_vocab.max_length + 2
        self.target_max_length = target_vocab.max_length + 2

    @abstractmethod
    def inference(self, texts: List[str]) -> List[str]:
        pass


class TranslateNN(nn.Module, TranslateModel, ABC):
    def __init__(self, source_vocab: Vocabulary, target_vocab: Vocabulary):
        nn.Module.__init__(self)
        TranslateModel.__init__(self, source_vocab, target_vocab)
        ABC.__init__(self)

    @abstractmethod
    def adapt_input(self, source_ids: torch.Tensor, source_lengths: torch.Tensor,
                    target_ids: torch.Tensor, target_lengths: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def loss(self, output: torch.Tensor, target_ids: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
