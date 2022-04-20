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
import json
from pathlib import Path

from utils import load_prepared_datasets, load_config
from evaluation import evaluate
from bert_utils import get_dataset, get_tokenized_dataset, get_tf_dataset

config = load_config()

RESULTS_PATH = config['RESULTS_PATH']
JOBS_PATH = config['JOBS_PATH']


class BERT():

    def __init__(self, params):
        self.params = params

        self.model_id = params['model_id']
        self.dataset_id = params['dataset_id']
        self.batch_size = params['batch_size']
        self.freeze_bert_encoder = params['freeze_bert_encoder']
        self.load_checkpoint_from = params['load_checkpoint_from']
        self.save_checkpoints = params['save_checkpoints']
        self.epochs = params['epochs']
        self.save_results = params['save_results']

        # Create tokenizer and load model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        config = AutoConfig.from_pretrained(self.model_id, num_labels=5)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_id, config=config, from_pt=True)
        self.model.bert.trainable = False
        self.model.bert.encoder.trainable = not self.freeze_bert_encoder
        self.model.compile(optimizer='adam',
                           loss=SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.summary()

        # Generate a unique ID for this run
        self.model_short = self.model_id.split('/')[-1]
        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        self.id = f'{self.model_short}_{self.dataset_id}_{timestamp}'

        # Create results directory
        Path(os.path.join(RESULTS_PATH, self.id)).mkdir(parents=True,
                                                        exist_ok=True)

        # Load weights from checkpoint if specified
        if self.load_checkpoint_from:
            self.model.load_weights(self.load_checkpoint_from)

        # Save the parameters
        with open(os.path.join(RESULTS_PATH, self.id, 'params.txt'), 'w') as f:
            params = json.dump(self.params, f, indent=4)

    def load_data(self):
        self.dataset = get_dataset(self.dataset_id)
        self.tokenized_dataset = get_tokenized_dataset(self.dataset_id,
                                                       self.dataset,
                                                       self.tokenizer)
        self.tf_dataset = get_tf_dataset(self.tokenized_dataset,
                                         self.batch_size, self.tokenizer)

    def train(self):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(RESULTS_PATH, self.id, 'checkpoint'))
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=0, restore_best_weights=True)

        self.model.fit(x=self.tf_dataset['train'],
                       validation_data=self.tf_dataset['valid'],
                       epochs=self.epochs,
                       callbacks=[checkpoint_callback, early_stopping_callback]
                       if self.save_checkpoints else [early_stopping_callback])

    def evaluate(self):
        preds = self.model.predict(self.tf_dataset['test'])[0]
        y_true = np.array(self.dataset['test']['label'])
        evaluate(self.id, preds, y_true, save_results=self.save_results)


if __name__ == '__main__':
    from datasets import Dataset, load_dataset
    from transformers import DataCollatorWithPadding
    from sklearn.model_selection import train_test_split
    import json
    import argparse

    from utils import set_seeds

    parser = argparse.ArgumentParser(description='Point to a job.')
    parser.add_argument('-j', '--job', action='store')
    args = parser.parse_args()

    set_seeds()

    job = args.job

    with open(job, 'r') as f:
        params = json.load(f)

    bert = BERT(params=params)
    bert.load_data()
    bert.train()
    bert.evaluate()
