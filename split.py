import json
import os
from typing import (
    List,
    Tuple,
)

import datasets
from sklearn.model_selection import GroupShuffleSplit

RANDOM_SEED = 42


def split_data(data: datasets.dataset_dict.DatasetDict, random_state=None) -> Tuple[List[int], List[int]]:
    gs_split = GroupShuffleSplit(n_splits=1,
                                 test_size=len(data['validation']) / len(data['train']),
                                 random_state=random_state)
    train_df = data['train'].to_pandas()
    for train_idx, val_idx in gs_split.split(X=train_df, groups=train_df.title):
        return train_idx.tolist(), val_idx.tolist()


def main():
    squad2 = datasets.load_dataset('squad_v2')
    train_idx, val_idx = split_data(squad2, random_state=RANDOM_SEED)
    os.makedirs('./finetuned', exist_ok=True)
    with open('./finetuned/train_idx.json', 'w') as f:
        json.dump(train_idx, f)
    with open('./finetuned/val_idx.json', 'w') as f:
        json.dump(val_idx, f)


if __name__ == '__main__':
    main()
