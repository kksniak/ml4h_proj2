from datasets import Dataset, DatasetDict, load_metric
from sklearn.utils import compute_class_weight
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_scheduler, BertForSequenceClassification, TrainerCallback
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm
from datetime import datetime
import os

from utils import load_all_datasets
from evaluation import evaluate


class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(
            [2.24909476, 0.61220699, 1.30219008, 0.57730516, 2.37068504]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class SaveCallback(TrainerCallback):

    def __init__(self, bert):
        self.bert = bert

    def on_epoch_end(self, args, state, control, **kwargs):
        self.bert.save()


class Bert:

    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=5)

        for param in self.model.bert.parameters():
            param.requires_grad = False

        self.training_args = TrainingArguments(
            output_dir="./logs",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
        )
        self.id = str(datetime.now())

    def _preprocess(self, df):
        df = df.rename(columns={'target': 'label'})

        label_map = {
            'BACKGROUND': 0,
            'METHODS': 1,
            'CONCLUSIONS': 2,
            'RESULTS': 3,
            'OBJECTIVE': 4
        }

        df['label'] = df['label'].map(label_map)

        return Dataset.from_pandas(df)

    def preprocess(self, train, valid, test, debug=False):
        train = self._preprocess(train)
        valid = self._preprocess(valid)
        test = self._preprocess(test)

        # Use a subset for performance reasons
        train = train.shuffle(seed=42).select(range(100000))

        if debug:
            train = train.shuffle(seed=42).select(range(20))
            valid = valid.shuffle(seed=42).select(range(20))
            test = test.shuffle(seed=42).select(range(20))

        dataset = DatasetDict()
        dataset['train'] = train
        dataset['valid'] = valid
        dataset['test'] = test

        tokenized_datasets = dataset.map(lambda sample: self.tokenizer(
            sample['text'],
            truncation=True,
        ))

        return tokenized_datasets

    def train(self, tokenized_datasets):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        metric = load_metric("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            self.save()
            return metric.compute(predictions=predictions, references=labels)

        self.trainer = CustomTrainer(model=self.model,
                                     args=self.training_args,
                                     train_dataset=tokenized_datasets["train"],
                                     eval_dataset=tokenized_datasets["test"],
                                     tokenizer=self.tokenizer,
                                     data_collator=data_collator,
                                     compute_metrics=compute_metrics,
                                     callbacks=[SaveCallback(self)])

        self.trainer.train()

    def save(self, path=None):
        if path is None:
            path = f'checkpoints/{self.id}'
        self.trainer.save_model(path)

    def load(self, path=None):

        if path is None:
            filename = os.listdir('checkpoints')[-1]
            path = f'checkpoints/{filename}'

        self.model = BertForSequenceClassification.from_pretrained(path)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

    def predict(self, dataset):
        return self.trainer.predict(dataset)[0]


if __name__ == '__main__':

    train, valid, test = load_all_datasets()

    bert = Bert()

    tokenized_datasets = bert.preprocess(train, valid, test, debug=False)

    # Train a new model
    bert.train(tokenized_datasets)

    # Load a saved model
    # bert.load()

    # Evaluation
    preds = bert.predict(tokenized_datasets['test'])

    y_true = np.array(tokenized_datasets['test']['label'])

    evaluate('test', preds, y_true, save_results=True)
