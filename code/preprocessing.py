from xmlrpc.client import boolean
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import string
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from utils import load_all_datasets


class Preprocessing:

    def __init__(self,
                 stemming: boolean = False,
                 lemmatisation: boolean = False) -> None:
        self.if_stemming = stemming
        self.if_lemmatisation = lemmatisation
        self.train_df, self.val_df, self.test_df = load_all_datasets()

    def preprocess_datasets(
            self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # print(self.test_df)

        self.lowercasing()
        self.replace_digits()
        self.stop_words_punctuation_removal()

        if self.if_stemming:
            self.apply_stemming()
        elif self.if_lemmatisation:
            self.apply_lemmatisation()

        #self.tokenisation()
        # print(self.test_df)
        return self.train_df, self.val_df, self.test_df

    def get_X_and_encoded_Y(
        self
    ) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
        label_encoder = LabelEncoder()

        sentences_train = self.train_df["text"].tolist()
        sentences_val = self.val_df["text"].tolist()
        sentences_test = self.test_df["text"].tolist()

        y_train = label_encoder.fit_transform(
            self.train_df["target"].to_numpy())
        y_val = label_encoder.transform(self.val_df["target"].to_numpy())
        y_test = label_encoder.transform(self.test_df["target"].to_numpy())

        return sentences_train, sentences_val, sentences_test, y_train, y_val, y_test

    def lowercasing(self) -> None:
        self.train_df["text"] = self.train_df["text"].str.lower()
        self.val_df["text"] = self.val_df["text"].str.lower()
        self.test_df["text"] = self.test_df["text"].str.lower()

    def stop_words_punctuation_removal(self) -> None:
        english_stop_words = set(nltk.corpus.stopwords.words('english'))

        def removal(text):
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            return " ".join([
                word for word in str(text).split()
                if word not in english_stop_words
            ])

        self.train_df["text"] = self.train_df["text"].apply(
            lambda text: removal(text))
        self.val_df["text"] = self.val_df["text"].apply(
            lambda text: removal(text))
        self.test_df["text"] = self.test_df["text"].apply(
            lambda text: removal(text))

    def replace_digits(self) -> None:

        def replace(text):
            return re.sub(r'[0-9]+', '@', text)

        self.train_df["text"] = self.train_df["text"].apply(
            lambda text: replace(text))
        self.val_df["text"] = self.val_df["text"].apply(
            lambda text: replace(text))
        self.test_df["text"] = self.test_df["text"].apply(
            lambda text: replace(text))

    def apply_stemming(self) -> None:
        stemmer = nltk.stem.PorterStemmer()

        def ap_stem(text):
            return " ".join([stemmer.stem(word) for word in text.split()])

        self.train_df['text'] = self.train_df['text'].apply(
            lambda text: ap_stem(text))
        self.val_df['text'] = self.val_df['text'].apply(
            lambda text: ap_stem(text))
        self.test_df['text'] = self.test_df['text'].apply(
            lambda text: ap_stem(text))

    def apply_lemmatisation(self) -> None:
        lemmatiser = nltk.stem.WordNetLemmatizer()

        def ap_lemm(text):
            return " ".join(
                [lemmatiser.lemmatize(word) for word in text.split()])

        self.train_df['text'] = self.train_df['text'].apply(
            lambda text: ap_lemm(text))
        self.val_df['text'] = self.val_df['text'].apply(
            lambda text: ap_lemm(text))
        self.test_df['text'] = self.test_df['text'].apply(
            lambda text: ap_lemm(text))

    def tokenisation(self) -> None:

        def token(text):
            return [word for word in str(text).split()]

        self.train_df["text"] = self.train_df["text"].apply(
            lambda text: token(text))
        self.val_df["text"] = self.val_df["text"].apply(
            lambda text: token(text))
        self.test_df["text"] = self.test_df["text"].apply(
            lambda text: token(text))


if __name__ == '__main__':
    dataset_preprocessing = Preprocessing(stemming=False, lemmatisation=False)
    dataset_preprocessing.preprocess_datasets()
    sentences_train, sentences_val, sentences_test, y_train, y_val, y_test = dataset_preprocessing.get_X_and_encoded_Y(
    )

    print(sentences_train[0])
