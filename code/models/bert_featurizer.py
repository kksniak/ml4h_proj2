import numpy as np
import os
import pickle
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, TFBertModel, DataCollatorWithPadding, Trainer, TrainingArguments

from bert_utils import get_dataset, get_tokenized_dataset, get_tf_dataset, get_tf_split
from utils import load_config

from typing import Literal

SHARDS = 100


def generate_features(model_id: str,
                      dataset_id: Literal['full', 'small_balanced', 'small',
                                          'mini', 'debug'],
                      split: Literal['train', 'valid', 'test']):
    """Generates features using a specified BERT model.

    Saves the output of the pooled output layer of the BERT model.

    Args:
        model_id: huggingface id of model.
        dataset_id: dataset to featurize.
        split: the split of the dataset to featurize.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    model = TFBertModel.from_pretrained(model_id, config=config, from_pt=True)

    dataset = get_dataset(dataset_id)
    tokenized_dataset = get_tokenized_dataset(dataset_id,
                                              dataset,
                                              tokenizer,
                                              pad=True,
                                              use_cache=False)

    y = np.array(tokenized_dataset[split]['label'])

    identifier = f'{model_id.split("/")[-1]}_{dataset_id}_{split}'
    config = load_config()
    with open(os.path.join(config['DATA_PATH'], f'y_{identifier}.npy'),
              'wb') as f:
        np.save(f, y)

    # Process data in shards to avoid filling GPU memory
    Path(os.path.join(config['DATA_PATH'], 'shards')).mkdir(parents=True,
                                                            exist_ok=True)
    for i in range(SHARDS):
        print('Processing shard', i)
        shard = tokenized_dataset[split].shard(num_shards=SHARDS, index=i)
        tf_shard = get_tf_split(shard, 1, tokenizer)
        X = model.predict(tf_shard, verbose=1)[1]
        with open(
                os.path.join(config['DATA_PATH'], 'shards',
                             f'X_feat_{identifier}-{i}.npy'), 'wb') as f:
            np.save(f, X)


if __name__ == '__main__':
    model_id = 'emilyalsentzer/Bio_ClinicalBERT'
    dataset_id = 'debug'

    generate_features(model_id, dataset_id, 'train')
    generate_features(model_id, dataset_id, 'test')
