import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import datasets
import transformers.pipelines.base
from datasets import load_dataset, load_from_disk
from transformers import pipeline


def get_predictions_with_pipeline(
        pipe: transformers.pipelines.base.Pipeline,
        dataset: datasets.arrow_dataset.Dataset,
        batch_size: int = 8,
        no_answer_threshold: Optional[float] = None
):
    enforce_threshold = no_answer_threshold is not None
    ids = dataset['id']
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names
         if col not in ['context', 'question']])
    outs = pipe(dataset, batch_size=batch_size)

    predictions = {}
    na_probs = {}
    for id_, out in zip(ids, outs):
        if enforce_threshold:
            predictions[id_] = out['answer'] if 1 - out['score'] < no_answer_threshold else ''
            na_probs[id_] = float(1 - out['score'] >= no_answer_threshold)
        else:
            predictions[id_] = out['answer']
            na_probs[id_] = 1 - out['score']
    return predictions, na_probs


def get_predictions_pretrained(
        model_name: str,
        dataset: datasets.arrow_dataset.Dataset,
        batch_size: int = 8,
        device: Optional[torch.device] = None,
        no_answer_threshold: Optional[float] = None
):
    if device is None:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    qa_pipe = pipeline('question-answering',
                       model=model_name,
                       tokenizer=model_name,
                       device=device)
    return get_predictions_with_pipeline(
        pipe=qa_pipe,
        dataset=dataset,
        batch_size=batch_size,
        no_answer_threshold=no_answer_threshold,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--no_answer_threshold', type=float)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pred_path', type=str, default='pred.json')
    parser.add_argument('--na_prob_path', type=str, default='na_prob.json')

    parameters = parser.parse_args()
    if parameters.data_path is not None:
        dataset = load_from_disk(parameters.data_path)
    else:
        dataset = load_dataset('squad_v2')['validation']

    predictions, na_probs = get_predictions_pretrained(
        model_name=parameters.model_name,
        dataset=dataset,
        batch_size=parameters.batch_size,
        no_answer_threshold=parameters.no_answer_threshold,
    )
    os.makedirs(Path(parameters.pred_path).parent.resolve(), exist_ok=True)
    with open(parameters.pred_path, 'w') as f:
        json.dump(predictions, f)
    os.makedirs(Path(parameters.pred_path).parent.resolve(), exist_ok=True)
    with open(parameters.na_prob_path, 'w') as f:
        json.dump(na_probs, f)


if __name__ == '__main__':
    main()
