from typing import Literal, Tuple, List
from xmlrpc.client import boolean
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten,MaxPooling1D

import sys

sys.path.append("../")

from utils import set_seeds


class ResNet1D_model:

    def __init__(self,
                 embedding_type: Literal["word2vec", "fastText", "kerasEmbed"],
                 embedding_layers: List[keras.layers.Layer] = []):
        self.embedding_type = embedding_type
        self.embedding_layers = embedding_layers

    def init_model(self, input_shape: Tuple):
        inputs = keras.Input(shape=input_shape)
        x = inputs

        for layer in self.embedding_layers:
            x = layer(x)

        x = Conv1D(10, 12, activation="relu")(x)
       
        x = Conv1D(10, 9, activation="relu")(x)
        x = Conv1D(20, 9, activation="relu")(x) 
        block1_out = MaxPooling1D(2,2)(x)

        x = Conv1D(20, 9, activation="relu",padding="same")(block1_out)
        x = Conv1D(30, 9, activation="relu",padding="same")(x)

        resized_block1_out = Conv1D(30,1, activation="relu")(block1_out)
        block2_out = keras.layers.add([x,resized_block1_out])

        x = Conv1D(30, 9, activation="relu",padding="same")(block2_out)
        x = Conv1D(30, 9, activation="relu",padding="same")(x)

        block3_out = keras.layers.add([x,block2_out])

        x = Flatten()(block3_out)
        outputs = Dense(5, activation="softmax")(x)

        self.clf = keras.Model(inputs, outputs)
        self.clf.summary()

    def train(self, X_train: np.array, y_train: np.array, X_val: np.array,
              y_val: np.array, load_model: boolean):

        if load_model:
            print("Loading model...")
            self.clf = keras.models.load_model(
                "code/models/models_checkpoints/ResNet_1D_" + self.embedding_type)
            return

        print("Fitting model...")
        set_seeds()
        self.init_model(input_shape=(X_train.shape[-1]))

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
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
        self.clf.save("code/models/models_checkpoints/ResNet_1D_" +
                      self.embedding_type)

    def predict(self, X_test):
        print("Making predictions...")
        return self.clf.predict(X_test)
