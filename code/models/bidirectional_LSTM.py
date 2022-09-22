from typing import Literal, Tuple, List
from xmlrpc.client import boolean
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Bidirectional, LSTM

from config import MODEL_CHECKPOINTS_PATH

from utils import set_seeds


class BiRNN_LSTM:

    def __init__(self,
                 embedding_type: Literal["word2vec", "fastText", "kerasEmbed"],
                 embedding_layers: List[keras.layers.Layer] = []):
        """Bidirectional LSTM model.

        Args:
            embedding_type: Name of the used embedding.
            embedding_layers: Embedding layer to use.
        """
        self.embedding_type = embedding_type
        self.embedding_layers = embedding_layers

    def init_model(self, input_shape: Tuple):
        """Definies model.

        Args:
            input_shape: Shape of the input as required by keras.
        """
        inputs = keras.Input(shape=input_shape)
        x = inputs

        for layer in self.embedding_layers:
            x = layer(x)

        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = Bidirectional(LSTM(32))(x)
        x = Dense(20, activation="relu")(x)
        outputs = Dense(5, activation="softmax")(x)

        self.clf = keras.Model(inputs, outputs)
        self.clf.summary()

    def train(self, X_train: np.array, y_train: np.array, X_val: np.array,
              y_val: np.array, load_model: boolean):

        if load_model:
            print("Loading model...")
            self.clf = keras.models.load_model(
                f"{MODEL_CHECKPOINTS_PATH}/biRNN_" + self.embedding_type)
            return

        print("Fitting model...")
        set_seeds()
        self.init_model(input_shape=(X_train.shape[-1],) )

        callback = tf.keras.callbacks.EarlyStopping(patience=3)

        self.clf.compile("adam",
                         "sparse_categorical_crossentropy",
                         metrics=["accuracy"])
        self.clf.fit(X_train,
                     y_train,
                     batch_size=512,
                     epochs=10,
                     validation_data=(X_val, y_val),
                     callbacks=[callback])

        print("Saving model...")
        self.clf.save(f"{MODEL_CHECKPOINTS_PATH}/biRNN_" +
                      self.embedding_type)

    def predict(self, X_test):
        print("Making predictions...")
        return self.clf.predict(X_test)
