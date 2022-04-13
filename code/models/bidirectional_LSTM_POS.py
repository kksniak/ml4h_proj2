from typing import Literal, Tuple, List
from xmlrpc.client import boolean
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Bidirectional, LSTM

import sys

sys.path.append("../")

from utils import set_seeds


class BiRNN_LSTM_POS:

    def __init__(self,
                 embedding_type: Literal["word2vec", "fastText", "kerasEmbed"],
                 embedding_layers: List[keras.layers.Layer] = []):
        self.embedding_type = embedding_type
        self.embedding_layers = embedding_layers

    def init_model(self, input_shape: Tuple, pos_shape: Tuple):
        inputs = keras.Input(shape=input_shape, name="sentences")
        x = inputs
        pos_input = keras.Input(shape = pos_shape, name="pos")
        for layer in self.embedding_layers:
            x = layer(x)

        
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = Bidirectional(LSTM(32))(x)

        pos_x = Dense(30,activation="relu")(pos_input)
        concatted = keras.layers.Concatenate()([x,pos_x])

        x = Dense(20, activation="relu")(concatted)
        outputs = Dense(5, activation="softmax")(x)


        self.clf = keras.Model(inputs= [inputs,pos_input], outputs = outputs)
        self.clf.summary()

    def train(self, X_train: np.array, y_train: np.array, X_val: np.array,
              y_val: np.array, pos_train:np.array, pos_val:np.array, load_model: boolean):

        if load_model:
            print("Loading model...")
            self.clf = keras.models.load_model(
                "code/models/models_checkpoints/biRNN_" + self.embedding_type)
            return

        print("Fitting model...")
        set_seeds()
        self.init_model(input_shape=(X_train.shape[-1],), pos_shape=(pos_train.shape[-1],))

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        self.clf.compile("adam",
                         "sparse_categorical_crossentropy",
                         metrics=["accuracy"])
        self.clf.fit({"sentences": X_train,"pos":pos_train},
                     y_train,
                     batch_size=512,
                     epochs=10,
                     validation_data=([X_val, pos_val], y_val),
                     callbacks=[callback])

        print("Saving model...")
        self.clf.save("code/models/models_checkpoints/biRNN_" +
                      self.embedding_type)

    def predict(self, X_test:np.array, pos_test:np.array):
        print("Making predictions...")
        return self.clf.predict({"sentences": X_test,"pos":pos_test})
