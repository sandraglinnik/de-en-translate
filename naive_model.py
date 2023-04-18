from typing import Optional, List

import torch

from model_base import TranslateModel
from dataset import TextDataset
from vocab import Vocabulary


class NaiveWordByWordModel(TranslateModel):
    def __init__(self, source_vocab: Vocabulary, target_vocab: Vocabulary) -> None:
        super().__init__(source_vocab, target_vocab)
        self.translations: Optional[torch.Tensor] = None

    def fit(self, dataset: TextDataset) -> None:
        counts: torch.Tensor = torch.zeros(self.source_vocab.size, self.target_vocab.size, dtype=torch.int)

        for source_ids, sl, target_ids, tl in dataset:
            length = min(sl, tl)
            for a, b in zip(source_ids[:length], target_ids[:length]):
                counts[a][b] += 1
        self.translations = torch.argmax(counts, dim=1)

    def inference(self, texts: List[str]) -> List[str]:
        assert self.translations is not None, 'model is not fitted yet'
        return [self.target_vocab.decode(self.translations[self.source_vocab.encode(text)].tolist()) for text in texts]
