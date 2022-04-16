from transformers import AutoTokenizer, AutoConfig, TFAutoModelForSequenceClassification, pipeline
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
from sklearn.utils import compute_class_weight


class BERT():

    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name, num_labels=5)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name, config=config)
        self.model.layers[0].trainable = False
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def _preprocess(self, df):
        df = df.rename(columns={'target': 'label'})

        label_map = {
            'BACKGROUND': np.array([1, 0, 0, 0, 0]),
            'METHODS': np.array([0, 1, 0, 0, 0]),
            'CONCLUSIONS': np.array([0, 0, 1, 0, 0]),
            'RESULTS': np.array([0, 0, 0, 1, 0]),
            'OBJECTIVE': np.array([0, 0, 0, 0, 1]),
        }

        df['label'] = df['label'].map(label_map)

        train_labels = df['label'].map(lambda l: np.argmax(l)).to_numpy()
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
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
            lambda sample: bert.tokenizer(sample['text'], truncation=True))

        data_collator = DataCollatorWithPadding(tokenizer=bert.tokenizer,
                                                return_tensors='tf')

        return {
            k: v.to_tf_dataset(
                columns=['attention_mask', 'input_ids', 'token_type_ids'],
                label_cols=['label'],
                shuffle=True,
                collate_fn=data_collator,
                batch_size=8,
            ) for k, v in tokenized_dataset.items()
        }, class_weights 



if __name__ == '__main__':
    from datasets import Dataset, load_dataset
    from transformers import DataCollatorWithPadding
    from utils import load_all_datasets

    bert = BERT(model_name='cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

    train, valid, test = load_all_datasets()

    datasets, class_weights = bert.preprocess(train, valid, test, small=False, half=True)

    bert.model.summary()
    bert.model.fit(x=datasets['train'],
                   validation_data=datasets['valid'],
                   epochs=3,
                   class_weight=class_weights)
