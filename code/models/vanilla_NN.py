from typing import Literal, Tuple, List
from xmlrpc.client import boolean
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten

import sys
sys.path.append("../")

from utils import set_seeds


class Vanilla_NN:

    def __init__(self,  embedding_type: Literal["TF-IDF", "word2vec", "fastText", "kerasEmbed"], embedding_layers: List[keras.layers.Layer] = []):
        self.embedding_type = embedding_type
        self.embedding_layers = embedding_layers

    def init_model(self, input_shape: Tuple):
        inputs = keras.Input(shape = input_shape)
        x = inputs
        
        for layer in self.embedding_layers:
            x = layer(x)
        x = Flatten()(x)
        x = Dense(64, activation="sigmoid")(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation="sigmoid")(x)
        x = Dropout(0.2)(x)
        x = Dense(10, activation="sigmoid")(x)
        outputs = Dense(5, activation="softmax")(x)
        
        self.clf = keras.Model(inputs, outputs)
        self.clf.summary()

    def train(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, load_model: boolean):
        
        if load_model:
            print("Loading model...")
            self.clf = keras.models.load_model("code/models/models_checkpoints/vanila_NN_" + self.embedding_type)
            return        

        print("Fitting model...")
        set_seeds()
        self.init_model(input_shape=(X_train.shape[-1]))

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        self.clf.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        self.clf.fit(X_train, y_train, batch_size=512, epochs=20, validation_data=(X_val, y_val), callbacks=[callback])
        
        print("Saving model...")
        self.clf.save("code/models/models_checkpoints/vanila_NN_" + self.embedding_type)        

    def predict(self, X_test):
        print("Making predictions...")
        return self.clf.predict(X_test)