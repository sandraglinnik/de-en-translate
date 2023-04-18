from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from vocab import Vocabulary


class TextDataset(Dataset):
    def __init__(self, source_data_file: str,
                 source_vocab: Vocabulary,
                 target_data_file: str,
                 target_vocab: Vocabulary,
                 max_size: int = -1
                 ) -> None:
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        with open(source_data_file) as f:
            self.source_texts: List[str] = [line.strip() for line in f.readlines()]
        self.source_ids, self.source_lengths = zip(
            *[self.source_vocab.encode_with_specials(text) for text in self.source_texts])

        with open(target_data_file) as f:
            self.target_texts: List[str] = [line.strip() for line in f.readlines()]
        self.target_ids, self.target_lengths = zip(
            *[self.target_vocab.encode_with_specials(text) for text in self.target_texts])

        assert len(self.source_texts) == len(self.target_texts)

        self.size = max_size if max_size != -1 else len(self.source_ids)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int, torch.Tensor, int]:
        return torch.tensor(self.source_ids[item]), self.source_lengths[item], \
               torch.tensor(self.target_ids[item]), self.target_lengths[item]
