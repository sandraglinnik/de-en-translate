from typing import Tuple

import torch
from torch.utils.data import DataLoader

from context_lstm import ContextLSTM
from dataset import TextDataset
from make_translation import make_file_translation
from naive_model import NaiveWordByWordModel
from train import train
from transformer import Transformer
from vocab import Vocabulary

DATA_DIR = 'data'
device = torch.device(
    'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))


def get_vocabs_and_loaders(vocab_size=10000,
                           train_max_size=-1,
                           batch_size=64
                           ) -> Tuple[Vocabulary, Vocabulary, DataLoader, DataLoader]:
    source_vocab = Vocabulary(DATA_DIR + '/train.de-en.de', vocab_size)
    target_vocab = Vocabulary(DATA_DIR + '/train.de-en.en', vocab_size)
    train_dataset = TextDataset(DATA_DIR + '/train.de-en.de', source_vocab, DATA_DIR + '/train.de-en.en', target_vocab,
                                train_max_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TextDataset(DATA_DIR + '/val.de-en.de', source_vocab, DATA_DIR + '/val.de-en.en', target_vocab)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return source_vocab, target_vocab, train_dataloader, val_dataloader


def context_lstm():
    source_vocab, target_vocab, train_dataloader, val_dataloader = get_vocabs_and_loaders()
    model = ContextLSTM(source_vocab, target_vocab).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    res = train(model, optimizer, scheduler, train_dataloader, val_dataloader, 20)
    make_file_translation(model, DATA_DIR + '/test1.de-en.de', 'res_context_lstm.txt')
    return res


def naive():
    source_vocab, target_vocab, train_dataloader, _ = get_vocabs_and_loaders(vocab_size=100000)

    model = NaiveWordByWordModel(source_vocab, target_vocab)
    model.fit(train_dataloader.dataset)
    make_file_translation(model, DATA_DIR + '/test1.de-en.de', 'res_naive.txt')


def transformer():
    source_vocab, target_vocab, train_dataloader, val_dataloader = get_vocabs_and_loaders()
    model = Transformer(source_vocab, target_vocab).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98))
    res = train(model, optimizer, None, train_dataloader, val_dataloader, 50)
    make_file_translation(model, DATA_DIR + '/test1.de-en.de', 'res_transformer.txt')
    return res
