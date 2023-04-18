import math
from typing import List

from model_base import TranslateModel


def make_translation(model: TranslateModel, texts: List[str], batch_size=64) -> List[str]:
    res: List[str] = []
    for i in range(math.ceil(len(texts) / batch_size)):
        res += model.inference(texts[i * batch_size: (i + 1) * batch_size])
    return res


def make_file_translation(model: TranslateModel, src_filepath: str, dst_filepath: str, batch_size=64) -> None:
    with open(src_filepath) as f:
        texts = [line.strip() for line in f.readlines()]
    res = make_translation(model, texts, batch_size)
    with open(dst_filepath, 'w') as f:
        print(*res, sep='\n', file=f)
