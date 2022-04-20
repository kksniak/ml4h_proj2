import numpy as np
import os
import pickle
from transformers import AutoTokenizer, AutoConfig, TFBertModel, DataCollatorWithPadding

from bert_utils import get_dataset, get_tokenized_dataset, get_tf_dataset
from utils import load_config

from typing import Literal


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
    tf_dataset = get_tf_dataset(tokenized_dataset, 1, tokenizer)

    print('Generating features...')
    X = model.predict(tf_dataset[split], verbose=1)[1]
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