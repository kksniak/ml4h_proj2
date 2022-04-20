from xmlrpc.client import boolean
from typing import Tuple
import pandas as pd
import numpy as np
import re
import string
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from utils import load_all_datasets


class Preprocessing:
    """class for data preprocessing"""

    def __init__(self,
                 stemming: boolean = False,
                 lemmatisation: boolean = False) -> None:
        """class initialisation

        Args:
            stemming (boolean, optional): If true, the stemming is added to preprocessing. Defaults to False.
            lemmatisation (boolean, optional): If true, the lemmitisation is added to preprocessing. Defaults to False.
        """
        self.if_stemming = stemming
        self.if_lemmatisation = lemmatisation
        self.train_df, self.val_df, self.test_df = load_all_datasets()

    def preprocess_datasets(
            self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """preprocessing workflow

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: dataframes with preprocessed sentences, positional features and class name
        """
        # print(self.test_df)

        self.lowercasing()
        self.stop_words_punctuation_removal()
        self.replace_digits()

        if self.if_stemming:
            self.apply_stemming()
        elif self.if_lemmatisation:
            self.apply_lemmatisation()

        #self.tokenisation()
        # print(self.test_df)
        return self.train_df, self.val_df, self.test_df

    def get_X_and_encoded_Y(
        self
    ) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array,
               np.array, np.array, np.array]:
        """Generates data to be fed for training and embedding functions

        Returns:
            Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array]: Returns sentences, encoded class number and positional feature for train, val, test sets
        """
        name_mapping = {
            'BACKGROUND': 0,
            'OBJECTIVE': 1,
            'METHODS': 2,
            'RESULTS': 3,
            'CONCLUSIONS': 4
        }

        sentences_train = self.train_df["text"].tolist()
        sentences_val = self.val_df["text"].tolist()
        sentences_test = self.test_df["text"].tolist()

        y_train = np.array([
            name_mapping[label] for label in self.train_df["target"].to_numpy()
        ])
        y_val = np.array(
            [name_mapping[label] for label in self.val_df["target"].to_numpy()])
        y_test = np.array([
            name_mapping[label] for label in self.test_df["target"].to_numpy()
        ])

        abstractPosFeat_train = np.array(
            self.train_df["relative_position"].tolist())
        abstractPosFeat_val = np.array(
            self.val_df["relative_position"].tolist())
        abstractPosFeat_test = np.array(
            self.test_df["relative_position"].tolist())

        return sentences_train, sentences_val, sentences_test, y_train, y_val, y_test, abstractPosFeat_train, abstractPosFeat_val, abstractPosFeat_test

    def lowercasing(self) -> None:
        """lowercases all letters"""
        self.train_df["text"] = self.train_df["text"].str.lower()
        self.val_df["text"] = self.val_df["text"].str.lower()
        self.test_df["text"] = self.test_df["text"].str.lower()

    def stop_words_punctuation_removal(self) -> None:
        """removes stop words and punctuation"""
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
        """replaces all numbers with '@' sign"""

        def replace(text):
            return re.sub(r'[0-9]+', '@', text)

        self.train_df["text"] = self.train_df["text"].apply(
            lambda text: replace(text))
        self.val_df["text"] = self.val_df["text"].apply(
            lambda text: replace(text))
        self.test_df["text"] = self.test_df["text"].apply(
            lambda text: replace(text))

    def apply_stemming(self) -> None:
        """applies stemming to words"""
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
        """applies lemmatisation to words"""
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
        """splits sentences into words"""
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
    sentences_train, sentences_val, sentences_test, y_train, y_val, y_test, abstractPosFeat_train, abstractPosFeat_val, abstractPosFeat_test = dataset_preprocessing.get_X_and_encoded_Y(
    )

    print(sentences_train[0], y_train[0], abstractPosFeat_train[0])
