from collections import Counter
from typing import List, Tuple


class Vocabulary:
    def __init__(self, filepath: str, max_size=10000) -> None:
        self.bos_id = 0
        self.eos_id = 1
        self.unk_id = 2
        self.pad_id = 3
        self.max_length = 0  # without specials

        token_count = Counter()
        with open(filepath) as f:
            for line in f.readlines():
                tokens = line.strip().split(' ')
                token_count.update(tokens)
                self.max_length = max(self.max_length, len(tokens))

        self.tokens: List[str] = ['<bos>', '<eos>', '<unk>', '<pad>']  # specials
        for token, _ in sorted(token_count.items(), key=lambda x: -x[1])[:max_size]:
            self.tokens.append(token)

        self.indices = {token: idx for idx, token in enumerate(self.tokens)}

        self.size = len(self.tokens)

    def encode(self, text: str) -> List[int]:
        return [self.indices.get(token, self.unk_id) for token in text.split(' ')]

    def encode_with_specials(self, text: str) -> Tuple[List[int], int]:
        encoded = self.encode(text)
        pad_size = max(0, self.max_length - len(encoded))
        padded = ([self.bos_id] + encoded + [self.eos_id] + [self.pad_id] * pad_size)[:self.max_length + 2]
        return padded, min(len(encoded), self.max_length) + 2

    def decode(self, indices: List[int], ignore_unk=True, ignore_eos=True) -> str:
        if not ignore_unk and self.unk_id in indices:
            indices = indices[:indices.index(self.unk_id)]
        if not ignore_eos and self.eos_id in indices:
            indices = indices[:indices.index(self.eos_id)]
        return ' '.join([self.tokens[idx] for idx in indices if idx >= 4])
