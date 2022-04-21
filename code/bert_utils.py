import os
import pickle
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding

from typing import Literal

from utils import load_prepared_datasets, load_config


def load_cached_dataset(cache_path: str) -> DatasetDict:
    """Loads a cached dataset from disk.

    Args:
        cache_path: path to the cached dataset.

    Returns:
        The loaded DatasetDict.
    """
    return DatasetDict.load_from_disk(cache_path)


def create_dataset(df: pd.DataFrame) -> Dataset:
    """Creates a huggingface Dataset from a pandas DataFrame.

    Args:
        df: dataframe containing text and labels.

    Returns:
        A huggingface dataset with integer labels.
    """

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
    """Loads a dataset from cache if exists and creates it otherwise.

    Caches the dataset if newly created.

    Args:
        dataset_id: dataset version to load.
        use_cache: Whether cache should be checked or not. Defaults to True.

    Returns:
        The loaded dataset.
    """

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

    # Create dataset if no cache was found
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
    """Creates a tokenized dataset from a dataset.

    Loads dataset from cache if exists, creates it otherwise.

    Args:
        dataset_id: dataset version to load/create.
        dataset: the dataset to tokenize.
        tokenizer: the tokenizer to use.
        pad: whether to pad the data or not. Defaults to False.
        use_cache: whether to load from cache. Defaults to True.

    Returns:
        The tokenized dataset.
    """

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
    """Creates a TensorFlow dataset from a tokenized dataset.

    Args:
        tokenized_dataset: the tokenized dataset.
        batch_size: batch size for the TF dataset.
        tokenizer: tokenizer to use for data collator.

    Returns:
        A TensorFlow dataset.
    """

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


def get_tf_split(tokenized_dataset, batch_size, tokenizer):
    """Creates a TensorFlow dataset from a single split of a tokenized dataset.

    Args:
        tokenized_dataset: the tokenized dataset.
        batch_size: batch size for the TF dataset.
        tokenizer: tokenizer to use for data collator.

    Returns:
        A TensorFlow dataset.
    """

    print('Creating TF dataset...')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                            return_tensors='tf')

    return tokenized_dataset.to_tf_dataset(
        columns=['attention_mask', 'input_ids', 'token_type_ids'],
        label_cols=['label'],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size,
    )