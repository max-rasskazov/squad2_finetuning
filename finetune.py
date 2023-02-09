from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
from copy import deepcopy
from pathlib import Path

import datasets
import numpy as np
import torch
import transformers
from torch import nn
from torch.optim import AdamW
from torch.optim.swa_utils import (
    SWALR,
    AveragedModel,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    get_cosine_schedule_with_warmup,
)


def fix_all_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_features(examples: dict,
                     tokenizer: transformers.PreTrainedTokenizer,
                     max_length: int,
                     doc_stride: int):
    """
    transforms datasets to features

    code adapted from the tutorial
    https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb
    """
    pad_on_right = tokenizer.padding_side == "right"
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offsets = tokenized_examples["offset_mapping"][i]

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else [-1, -1])
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


class FeatureDataset(Dataset):
    def __init__(self, features_hf_dataset: datasets.arrow_dataset.Dataset):
        self.features = features_hf_dataset

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item: int):
        return self.features[item]


def reinit_top_layers(model: nn.Module, n_layers: int):
    for layer in model.base_model.encoder.layer[-n_layers:]:
        layer.apply(model._init_weights)


def get_optimizer_grouped_parameters(*,
                                     model: transformers.PreTrainedModel,
                                     learning_rate: float,
                                     weight_decay: float,
                                     layerwise_learning_rate_decay: float,
                                     ) -> list[dict]:
    """
    makes parameter groups with LLRD for all the layers and embeddings

    code adapted from here
    https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning/notebook
    """
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    # initialize lrs for every layer
    layers = [model.base_model.embeddings] + list(model.base_model.encoder.layer)
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters


class ModelKeeper:
    def __init__(self, model: nn.Module,
                 checkpoint_path: os.PathLike | str,
                 checkpoint_id: int | None = None,
                 best_loss: float | None = None):
        model_device = model.device
        self.best_model = deepcopy(model.cpu().state_dict())
        model.to(model_device)
        self.best_loss = best_loss or float('inf')
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_id = checkpoint_id or 0
        os.makedirs(checkpoint_path, exist_ok=True)

    def make_checkpoint(self, model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        epoch: int | None = None,
                        loss: float | None = None) -> None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, self.checkpoint_path / f'{model.__class__.__name__}__{self.checkpoint_id}')
        self.checkpoint_id += 1

    def update(self,
               loss: float,
               model: nn.Module,
               optimizer: torch.optim.Optimizer,
               epoch: int | None = None):
        if loss < self.best_loss:
            self.best_loss = loss
            model_device = model.device
            self.best_model = deepcopy(model.cpu().state_dict())
            model.to(model_device)
            self.make_checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=loss)

    def load_checkpoint(self, checkpoint_id):
        return torch.load(self.checkpoint_path
                          / f'{self.best_model.__class__.__name__}__{checkpoint_id}')


def optimizer_to(optim, device):
    """
    https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/2
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def train_eval_epoch(model: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     scheduler: torch.optim.lr_scheduler._LRScheduler,
                     device: torch.device,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     writer: SummaryWriter,
                     epoch: int,
                     swa_model: torch.optim.swa_utils.AveragedModel,
                     swa_scheduler: torch.optim.swa_utils.SWALR,
                     swa_start: int,
                     model_keeper: ModelKeeper,
                     eval_per_epoch: int | None = None,
                     swa_updates_per_epoch: int | None = None,
                     ) -> None:
    model.to(device)
    optimizer_to(optimizer, device)
    n_batches = len(train_loader)
    eval_frequency = None
    if eval_per_epoch is not None and eval_per_epoch > 1:
        eval_frequency = n_batches // eval_per_epoch
    swa_frequency = None
    if swa_updates_per_epoch is not None and swa_updates_per_epoch > 1:
        swa_frequency = n_batches // swa_updates_per_epoch
    signature_columns = list(inspect.signature(model.forward).parameters.keys())
    model.train()
    for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}:', position=0), start=1):
        optimizer.zero_grad()
        out = model(**{k: v.to(device) for k, v in batch.items() if k in signature_columns})
        writer.add_scalar('Loss/train',
                          out['loss'].data.cpu().item(),
                          epoch * n_batches + i)
        out['loss'].backward()
        optimizer.step()

        if epoch >= swa_start:
            swa_scheduler.step()
        else:
            scheduler.step()

        if eval_frequency is not None and i % eval_frequency == 0:
            val_loss = evaluate(model, val_loader, device)
            writer.add_scalar('Loss/val', val_loss, epoch * n_batches + i)
            logging.info(f'{val_loss=}; {epoch=}; batch={epoch * n_batches + i}')
            model_keeper.update(val_loss, model, optimizer, epoch)
            model.train()
            writer.flush()
        if epoch >= swa_start and swa_frequency is not None and i % swa_frequency == 0:
            swa_model.update_parameters(model)

    if eval_frequency is None:
        val_loss = evaluate(model, val_loader, device)
        writer.add_scalar('Loss/val', val_loss, epoch * n_batches + n_batches)
        logging.info(f'{val_loss=:.4f}; {epoch=}; batch={epoch * n_batches + n_batches}')
        model_keeper.update(val_loss, model, optimizer, epoch)

    writer.flush()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    n = sum_loss = 0
    model.to(device)
    model.eval()
    signature_columns = list(inspect.signature(model.forward).parameters.keys())
    with torch.no_grad():
        for batch in loader:
            cur_batch_size = batch['input_ids'].shape[0]
            n += cur_batch_size
            out = model(**{k: v.to(device) for k, v in batch.items() if k in signature_columns})
            sum_loss += out['loss'].data * cur_batch_size
    return sum_loss.cpu().item() / n


def training_loop(*,
                  model: transformers.PreTrainedModel,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  test_loader: DataLoader,
                  top_layers_to_reinit: int,
                  learning_rate: float,
                  layerwise_learning_rate_decay: float,
                  weight_decay: float,
                  n_epoch: int,
                  swa_start: int,
                  swa_lr: float,
                  swa_anneal_epochs: int,
                  num_warmup_steps: int,
                  num_training_steps: int,
                  eval_per_regular_epoch: int,
                  swa_updates_per_epoch: int,
                  checkpoint: dict | None,
                  ) -> transformers.PreTrainedModel:
    # init part
    reinit_top_layers(model, top_layers_to_reinit)

    grouped_optimizer_params = get_optimizer_grouped_parameters(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        layerwise_learning_rate_decay=layerwise_learning_rate_decay,
    )
    optimizer = AdamW(
        grouped_optimizer_params,
        lr=learning_rate,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=swa_anneal_epochs)
    writer = SummaryWriter('runs/finetuned2')

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model_keeper = ModelKeeper(model,
                               checkpoint_path='finetuned/checkpoints',
                               best_loss=checkpoint['loss'] if checkpoint else None,
                               checkpoint_id=checkpoint['checkpoint_id'] + 1 if checkpoint else None,)
    n_batches_per_epoch = len(train_loader)

    for ep in tqdm(range(n_epoch), desc='epochs: '):
        train_eval_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            writer=writer,
            epoch=ep,
            swa_model=swa_model,
            swa_scheduler=swa_scheduler,
            swa_start=swa_start,
            model_keeper=model_keeper,
            eval_per_epoch=eval_per_regular_epoch,
            swa_updates_per_epoch=swa_updates_per_epoch,
        )

    best_model = model
    best_model.load_state_dict(model_keeper.best_model)
    test_loss_best = evaluate(best_model, test_loader, device)
    writer.add_scalar('Loss/test_best', test_loss_best, n_batches_per_epoch * n_epoch)
    logging.info(f'{test_loss_best=:.4f}; {n_epoch=}; batch={n_batches_per_epoch * n_epoch}')

    # take swa_model weights and evaluate the model:
    model.load_state_dict(swa_model.module.state_dict())
    val_loss_swa = evaluate(model, test_loader, device)
    writer.add_scalar('Loss/val_swa', val_loss_swa, n_batches_per_epoch * n_epoch)
    logging.info(f'{val_loss_swa=:.4f}; {n_epoch=}; batch={n_batches_per_epoch * n_epoch}')

    test_loss_swa = evaluate(model, test_loader, device)
    writer.add_scalar('Loss/test_swa', test_loss_swa, n_batches_per_epoch * n_epoch)
    logging.info(f'{test_loss_swa=:.4f}; {n_epoch=}; batch={n_batches_per_epoch * n_epoch}')

    model_keeper.make_checkpoint(model, optimizer, epoch=n_epoch, loss=val_loss_swa)

    writer.close()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_name', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--doc_stride', type=int, default=128)
    parser.add_argument('--top_layers_to_reinit', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--layerwise_learning_rate_decay', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--n_epoch', type=int, required=True)
    parser.add_argument('--swa_start', type=int, required=True)
    parser.add_argument('--swa_lr', type=float, required=True)
    parser.add_argument('--swa_anneal_epochs', type=int, required=True)
    parser.add_argument('--warmup_steps_share', type=float, required=True)
    parser.add_argument('--eval_per_regular_epoch', type=int, required=True)
    parser.add_argument('--swa_updates_per_epoch', type=int, required=True)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--random_seed', type=int, default=42)

    parameters = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s::%(name)s::%(asctime)s::%(message)s',
    )

    fix_all_seeds(parameters.random_seed)

    with open('./finetuned/train_idx.json') as f:
        train_idx = json.load(f)
    with open('./finetuned/val_idx.json') as f:
        val_idx = json.load(f)

    squad2 = datasets.load_dataset("squad_v2")
    squad2_split = datasets.dataset_dict.DatasetDict(
        train=squad2['train'].select(train_idx),
        val=squad2['train'].select(val_idx),
        test=squad2['validation'],
    )
    del squad2

    model = AutoModelForQuestionAnswering.from_pretrained(parameters.base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(parameters.base_model_name)

    squad2_features = squad2_split.map(
        prepare_features,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            max_length=parameters.max_length,
            doc_stride=parameters.doc_stride,
        ),
        batched=True,
        remove_columns=squad2_split["train"].column_names)
    del squad2_split

    train_ds = FeatureDataset(squad2_features['train'])
    val_ds = FeatureDataset(squad2_features['val'])
    test_ds = FeatureDataset(squad2_features['test'])

    train_loader = DataLoader(
        train_ds,
        batch_size=parameters.batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=torch.get_num_threads())
    val_loader = DataLoader(
        val_ds,
        batch_size=parameters.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=torch.get_num_threads())
    test_loader = DataLoader(
        test_ds,
        batch_size=parameters.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=torch.get_num_threads())

    n_batches_per_epoch = len(train_loader)
    regular_schedule_steps = parameters.swa_start * n_batches_per_epoch
    num_warmup_steps = int(parameters.warmup_steps_share * regular_schedule_steps)
    if parameters.checkpoint_path is not None:
        checkpoint = torch.load(parameters.checkpoint_path,
                                map_location=torch.device('cuda:0'
                                                          if torch.cuda.is_available()
                                                          else 'cpu'))
        checkpoint['checkpoint_id'] = int(
          parameters.checkpoint_path.split('_')[-1])
    else:
        checkpoint = None

    swa_model = training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        top_layers_to_reinit=parameters.top_layers_to_reinit,
        learning_rate=parameters.learning_rate,
        layerwise_learning_rate_decay=parameters.layerwise_learning_rate_decay,
        weight_decay=parameters.weight_decay,
        n_epoch=parameters.n_epoch,
        swa_start=parameters.swa_start,
        swa_lr=parameters.swa_lr,
        swa_anneal_epochs=parameters.swa_anneal_epochs,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=regular_schedule_steps,
        eval_per_regular_epoch=parameters.eval_per_regular_epoch,
        swa_updates_per_epoch=parameters.swa_updates_per_epoch,
        checkpoint=checkpoint,
    )
    swa_model.save_pretrained('finetuned/model')
    tokenizer.save_pretrained('finetuned/model')


if __name__ == '__main__':
    main()
