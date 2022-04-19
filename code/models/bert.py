from transformers import AutoTokenizer, AutoConfig, TFAutoModelForSequenceClassification, pipeline
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
from sklearn.utils import compute_class_weight
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import pickle
from datetime import datetime
import tensorflow as tf
from typing import Literal, Union
import os

from utils import load_prepared_datasets
from evaluation import evaluate

BASE_PATH = './'
CACHE_PATH = os.path.join(BASE_PATH, 'cache')
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'checkpoints')
JOBS_PATH = os.path.join(BASE_PATH, 'jobs')


class BERT():

    def __init__(self, params):
        self.params = params

        self.tokenizer = AutoTokenizer.from_pretrained(self.params['model'])
        config = AutoConfig.from_pretrained(self.params['model'], num_labels=5)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.params['model'], config=config, from_pt=True)
        self.model.layers[0].trainable = not self.params['freeze_base_layer']
        self.model.compile(optimizer='adam',
                           loss=SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.model_short = self.params['model'].split('/')[-1]
        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        self.id = f'{self.model_short}_{self.params["dataset"]}_{timestamp}'

        if self.params['load_checkpoint']:
            self.load_checkpoint()

    def load_checkpoint(self):
        if self.params['load_checkpoint'] is True:
            checkpoints = os.listdir(os.path.join(CHECKPOINT_PATH))
            checkpoints = [
                c for c in checkpoints
                if c.startswith(f'{self.model_short}_{self.params["dataset"]}')
            ]
            checkpoint = os.path.join(CHECKPOINT_PATH, checkpoints[-1])

        print('Loading checkpoint:', checkpoint)
        self.model.load_weights(checkpoint)

    def load_data(self):
        train, valid, test = load_prepared_datasets(
            variant=self.params['dataset'])
        self.tf_dataset, self.dataset, self.class_weights = self.preprocess(
            train,
            valid,
            test,
            restore_from_cache=self.params['load_cached_dataset'])

    def train(self):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_PATH, self.id))

        bert.model.summary()
        bert.model.fit(x=self.tf_dataset['train'],
                       validation_data=self.tf_dataset['valid'],
                       epochs=self.params['epochs'],
                       class_weight=self.class_weights,
                       callbacks=[checkpoint_callback]
                       if self.params['save_checkpoints'] else [])

    def _preprocess(self, df):
        df = df.rename(columns={'target': 'label'})

        label_map = {
            'BACKGROUND': 0,
            'METHODS': 1,
            'CONCLUSIONS': 2,
            'RESULTS': 3,
            'OBJECTIVE': 4,
        }

        df['label'] = df['label'].map(label_map)

        train_labels = df['label'].to_numpy()
        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=np.unique(train_labels),
                                             y=train_labels)
        class_weights = dict(enumerate(class_weights))

        return Dataset.from_pandas(df), class_weights

    def preprocess(self,
                   train,
                   valid,
                   test,
                   cache=True,
                   restore_from_cache=False):
        cache_path = os.path.join(CACHE_PATH, self.params['dataset'])
        should_preprocess = not restore_from_cache
        if restore_from_cache:
            try:
                dataset = DatasetDict.load_from_disk(
                    os.path.join(cache_path, 'dataset'))
                tokenized_dataset = DatasetDict.load_from_disk(
                    os.path.join(cache_path, 'tokenized_dataset'))
                with open(os.path.join(cache_path, 'class_weights.pkl'),
                          'rb') as f:
                    class_weights = pickle.load(f)
            except FileNotFoundError:
                print('No cache found, preprocessing dataset...')
                should_preprocess = True

        if should_preprocess:
            train, class_weights = self._preprocess(train)
            valid, _ = self._preprocess(valid)
            test, _ = self._preprocess(test)

            dataset = DatasetDict()
            dataset['train'] = train
            dataset['valid'] = valid
            dataset['test'] = test

            tokenized_dataset = dataset.map(
                lambda sample: self.tokenizer(sample['text'], truncation=True))

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer,
                                                return_tensors='tf')

        tf_dataset = {
            'train':
                tokenized_dataset['train'].to_tf_dataset(
                    columns=['attention_mask', 'input_ids', 'token_type_ids'],
                    label_cols=['label'],
                    shuffle=True,
                    collate_fn=data_collator,
                    batch_size=self.params['batch_size'],
                ),
            'valid':
                tokenized_dataset['valid'].to_tf_dataset(
                    columns=['attention_mask', 'input_ids', 'token_type_ids'],
                    label_cols=['label'],
                    shuffle=True,
                    collate_fn=data_collator,
                    batch_size=self.params['batch_size'],
                ),
            'test':
                tokenized_dataset['test'].to_tf_dataset(
                    columns=['attention_mask', 'input_ids', 'token_type_ids'],
                    label_cols=['label'],
                    shuffle=False,
                    collate_fn=data_collator,
                    batch_size=1,
                ),
        }

        if cache and should_preprocess:
            dataset.save_to_disk(os.path.join(cache_path, 'dataset'))
            tokenized_dataset.save_to_disk(
                os.path.join(cache_path, 'tokenized_dataset'))
            with open(os.path.join(cache_path, 'class_weights.pkl'), 'wb') as f:
                pickle.dump(class_weights, f)

        return tf_dataset, dataset, class_weights

    def evaluate(self):
        preds = self.model.predict(self.tf_dataset['test'])[0]
        y_true = np.array(self.dataset['test']['label'])
        evaluate(self.id,
                 preds,
                 y_true,
                 save_results=self.params['save_results'])


if __name__ == '__main__':
    from datasets import Dataset, load_dataset
    from transformers import DataCollatorWithPadding
    from sklearn.model_selection import train_test_split
    import json

    for job in os.listdir(JOBS_PATH):
        with open(os.path.join(JOBS_PATH, job), 'r') as f:
            params = json.load(f)

        bert = BERT(params=params)
        bert.load_data()
        bert.train()
        bert.evaluate()
