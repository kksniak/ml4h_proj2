from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, pipeline
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
from sklearn.utils import compute_class_weight


class BERT():

    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name, num_labels=5)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config)
        # self.model.layers[0].trainable = False
        # self.model.compile(optimizer='adam',
        #                    loss='categorical_crossentropy',
        #                    metrics=['accuracy'])

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

        train_labels = df['label'].map(lambda l: np.argmax(l)).to_numpy()
        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=np.unique(train_labels),
                                             y=train_labels)
        class_weights = dict(enumerate(class_weights))

        return Dataset.from_pandas(df), class_weights

    def preprocess(self, train, valid, test, small=False, half=False):
        train, class_weights = self._preprocess(train)
        valid, _ = self._preprocess(valid)
        test, _ = self._preprocess(test)

        dataset = DatasetDict()
        dataset['train'] = train
        dataset['valid'] = valid
        dataset['test'] = test

        if small:
            small_dataset = DatasetDict()
            small_dataset['train'] = dataset['train'].shuffle(seed=42).select(
                range(100))
            small_dataset['valid'] = dataset['valid'].shuffle(seed=42).select(
                range(20))
            small_dataset['test'] = dataset['test'].shuffle(seed=42).select(
                range(50))
            dataset = small_dataset
        elif half:
            half_dataset = DatasetDict()
            half_dataset['train'] = dataset['train'].shuffle(seed=42).select(
                range(500000))
            half_dataset['valid'] = dataset['valid'].shuffle(seed=42).select(
                range(20000))
            half_dataset['test'] = dataset['test'].shuffle(seed=42).select(
                range(20000))
            dataset = half_dataset

        tokenized_dataset = dataset.map(
            lambda sample: bert.tokenizer(sample['text'],
                                          padding='max_length',
                                          truncation=True,
                                          max_length=512))

        return tokenized_dataset, class_weights


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    from datasets import Dataset, load_dataset, load_metric
    from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
    from utils import load_all_datasets
    from evaluation import evaluate

    # bert = BERT(model_name='cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    bert = BERT(model_name='emilyalsentzer/Bio_ClinicalBERT')

    train, valid, test = load_all_datasets()

    datasets, class_weights = bert.preprocess(train,
                                              valid,
                                              test,
                                              small=True,
                                              half=False)

    training_args = TrainingArguments(output_dir="test_trainer",
                                      evaluation_strategy="epoch",
                                      num_train_epochs=1)
    metric = load_metric("accuracy")

    trainer = Trainer(
        model=bert.model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['valid'],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model()

    trainer

    y_pred = trainer.predict(datasets['test'])[0]

    y_true = np.array(datasets['test']['label'])

    evaluate('banan', y_pred, y_true, save_results=True)

    datasets['test']
    bert.model(datasets['test'])

    # bert.model.summary()
    # bert.model.fit(x=datasets['train'],
    #                validation_data=datasets['valid'],
    #                epochs=3,
    #                class_weight=class_weights)
