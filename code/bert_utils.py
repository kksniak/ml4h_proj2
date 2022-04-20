import os
import pickle
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding

from typing import Literal

from utils import load_prepared_datasets, load_config


def load_cached_dataset(cache_path):
    return DatasetDict.load_from_disk(cache_path)


def create_dataset(df: pd.DataFrame) -> Dataset:
    df = df.rename(columns={'target': 'label'})

    label_map = {
        'BACKGROUND': 0,
        'OBJECTIVE': 1,
        'METHODS': 2,
        'RESULTS': 3,
        'CONCLUSIONS': 4
    }

    df['label'] = df['label'].map(label_map)

    dataset = Dataset.from_pandas(df)

    return dataset


def get_dataset(dataset_id: Literal['full', 'small_balanced', 'small', 'mini',
                                    'debug'],
                use_cache=True):
    # Load data from cache if exists
    config = load_config()
    cache_path = os.path.join(config['DATA_CACHE_PATH'], dataset_id)
    dataset_path = os.path.join(cache_path, 'dataset')

    if use_cache:
        try:
            dataset = load_cached_dataset(dataset_path)
            print('Loaded dataset from cache')
            return dataset
        except FileNotFoundError:
            pass

    print('Creating dataset...')
    train, valid, test = load_prepared_datasets(variant=dataset_id)

    dataset = DatasetDict()
    dataset['train'] = create_dataset(train)
    dataset['valid'] = create_dataset(valid)
    dataset['test'] = create_dataset(test)

    dataset.save_to_disk(os.path.join(dataset_path))

    return dataset


def get_tokenized_dataset(dataset_id: Literal['full', 'small_balanced', 'small',
                                              'mini', 'debug'],
                          dataset: Dataset,
                          tokenizer,
                          pad=False,
                          use_cache=True):
    config = load_config()
    cache_path = os.path.join(config['DATA_CACHE_PATH'], dataset_id)
    tokenized_dataset_path = os.path.join(cache_path, 'tokenized_dataset')

    if use_cache:
        try:
            tokenized_dataset = load_cached_dataset(tokenized_dataset_path)
            print('Loaded tokenized dataset from cache')
            return tokenized_dataset
        except FileNotFoundError:
            pass

    print('Tokenizing dataset...')
    tokenized_dataset = dataset.map(
        lambda sample: tokenizer(sample['text'],
                                 truncation=True,
                                 padding='max_length' if pad else None,
                                 max_length=512 if pad else None))

    tokenized_dataset.save_to_disk(os.path.join(tokenized_dataset_path))

    return tokenized_dataset


def get_tf_dataset(tokenized_dataset, batch_size, tokenizer):
    print('Creating TF dataset...')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                            return_tensors='tf')

    tf_dataset = {}
    for key, dataset in tokenized_dataset.items():
        tf_dataset[key] = dataset.to_tf_dataset(
            columns=['attention_mask', 'input_ids', 'token_type_ids'],
            label_cols=['label'],
            shuffle=key != 'test',
            collate_fn=data_collator,
            batch_size=batch_size,
        )

    return tf_dataset