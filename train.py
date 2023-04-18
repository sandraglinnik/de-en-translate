from typing import List, Optional, Any, Tuple

from sacrebleu import BLEU
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from make_translation import make_translation
from model_base import TranslateNN


def bleu(predictions: List[str], target: List[str]) -> float:
    bleu = BLEU(tokenize=None)
    return bleu.corpus_score(predictions, [target]).score


def training_epoch(model: TranslateNN, optimizer: torch.optim.Optimizer, loader: DataLoader, tqdm_desc: str) -> float:
    train_loss = 0.0

    model.train()
    for source_ids, source_lens, target_ids, target_lens in tqdm(loader, desc=tqdm_desc):
        optimizer.zero_grad()
        output = model(*model.adapt_input(source_ids, source_lens, target_ids, target_lens))
        loss = model.loss(output, target_ids, target_lens)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(source_ids)

    train_loss /= len(loader.dataset)
    return train_loss


@torch.no_grad()
def validation_epoch(model: TranslateNN, loader: DataLoader, tqdm_desc: str) -> float:
    val_loss = 0.0

    model.eval()
    for source_ids, source_lens, target_ids, target_lens in tqdm(loader, desc=tqdm_desc):
        output = model(*model.adapt_input(source_ids, source_lens, target_ids, target_lens))
        loss = model.loss(output, target_ids, target_lens)
        val_loss += loss.item() * len(source_ids)

    val_loss /= len(loader.dataset)
    return val_loss


def train(model: TranslateNN,
          optimizer: torch.optim.Optimizer,
          scheduler: Optional[Any],
          train_loader: DataLoader,
          val_loader: DataLoader,
          num_epochs: int,
          compute_bleu: bool = False,
          log_wandb: bool = False
          ) -> Tuple[List[float], ...]:
    train_losses, val_losses = [], []
    val_bleus = []

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        log = {
            'train_loss': train_loss,
            'val_loss': val_loss
        }

        if compute_bleu:
            val_predictions = make_translation(model, val_loader.dataset.source_texts)
            val_bleu = bleu(val_predictions, val_loader.dataset.target_texts)
            log['val_bleu'] = val_bleu
            val_bleus.append(val_bleu)

        if scheduler is not None:
            scheduler.step()

        if log_wandb:
            wandb.log(log)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    if compute_bleu:
        return train_losses, val_losses, val_bleus
    return train_losses, val_losses
