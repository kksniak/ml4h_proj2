import numpy as np
import os
import pickle
from transformers import AutoTokenizer, AutoConfig, TFBertModel, DataCollatorWithPadding, Trainer, TrainingArguments

from bert_utils import get_dataset, get_tokenized_dataset, get_tf_dataset, get_tf_split
from utils import load_config

from typing import Literal

SHARDS = 100


def generate_features(model_id: str,
                      dataset_id: Literal['full', 'small_balanced', 'small',
                                          'mini', 'debug'],
                      split: Literal['train', 'valid', 'test']):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    model = TFBertModel.from_pretrained(model_id, config=config, from_pt=True)

    dataset = get_dataset(dataset_id)
    tokenized_dataset = get_tokenized_dataset(dataset_id,
                                              dataset,
                                              tokenizer,
                                              pad=True,
                                              use_cache=False)

    Xs = []
    for i in range(SHARDS):
        print('Processing shard', i)
        shard = tokenized_dataset[split].shard(num_shards=SHARDS, index=i)
        tf_shard = get_tf_split(shard, 1, tokenizer)
        X = model.predict(tf_shard, verbose=1)[1]
        Xs.append(X)

    X = np.concatenate(Xs)
    y = np.array(tokenized_dataset[split]['label'])

    config = load_config()

    identifier = f'{model_id.split("/")[-1]}_{dataset_id}_{split}'

    with open(os.path.join(config['DATA_PATH'], f'X_feat_{identifier}.npy'),
              'wb') as f:
        np.save(f, X)

    with open(os.path.join(config['DATA_PATH'], f'y_{identifier}.npy'),
              'wb') as f:
        np.save(f, y)


if __name__ == '__main__':
    model_id = 'emilyalsentzer/Bio_ClinicalBERT'
    dataset_id = 'debug'

    generate_features(model_id, dataset_id, 'train')
    generate_features(model_id, dataset_id, 'test')
