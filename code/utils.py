import os
import random
from tokenize import String
from typing import Tuple
import pandas as pd

import numpy as np
import tensorflow as tf
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import pathlib

SEED = 2137


def load_all_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # training
    train_df = load_dataset("data/train.txt")

    # validation
    val_df = load_dataset("data/dev.txt")

    # test
    test_df = load_dataset("data/test.txt")

    return train_df, val_df, test_df


def load_dataset(filename: String) -> pd.DataFrame:
    with open(filename, "r") as f:
        lines = f.readlines()

    sentences = list()
    position_features = list()
    sentence_number = 0

    for line in lines:
        # new abstract
        if line[:3] == "###":
            sentence_number = 0
            abstratct_id = int(line[3:])

        # new sentence
        elif len(line.strip()):
            sentence_number += 1
            data = dict()
            data["abstract_id"], data["target"], data[
                "text"] = abstratct_id, line.split("\t")[0], line.split("\t")[1]
            sentences.append(data)

        else:
            for i in range(sentence_number):
                data = dict()
                data['relative_position'] = [(i+1)/sentence_number, sentence_number]
                position_features.append(data)
    
    final_sentences = list()
    for i, data in enumerate(sentences):
        data.update(position_features[i])
        final_sentences.append(data)

    return pd.DataFrame(final_sentences)


def set_seeds() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    tf.random.set_seed(SEED)
    tf.keras.initializers.glorot_normal(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def create_POS_encoding(sentences:list, filename:str, vectorizer:TfidfVectorizer = None) -> TfidfVectorizer:
    nlp = spacy.load("en_core_web_sm")
    pos = []
    for  sentence in tqdm(sentences):
        doc = nlp(sentence)
        pos.append(" ".join([w.pos_ for w in doc]))

    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        X_tf = vectorizer.fit_transform(pos).toarray()

    else:
        X_tf = vectorizer.transform(pos).toarray()

    np.save(pathlib.Path(__file__).parents[1].joinpath(f"data/{filename}.npy"),X_tf)
    return vectorizer
