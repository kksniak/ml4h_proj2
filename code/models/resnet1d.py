from typing import Literal, Tuple, List
from xmlrpc.client import boolean
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten,MaxPooling1D

import sys

from config import MODEL_CHECKPOINTS_PATH

from utils import set_seeds


class ResNet1D_model:

    def __init__(self,
                 embedding_type: Literal["word2vec", "fastText", "kerasEmbed"],
                 embedding_layers: List[keras.layers.Layer] = [], use_len_and_position: bool = False):
        """ResNet1D model

        Args:
            embedding_type: Name of the used embedding.
            embedding_layers: Embedding layer to use.
            use_len_and_position: If model uses positional features. Defaults to False.
        """
        self.embedding_type = embedding_type
        self.embedding_layers = embedding_layers
        self.use_len_and_position = use_len_and_position


    def init_model(self, input_shape: Tuple,len_shape:Tuple = None):
        """Defines model.

        Args:
            input_shape: Shape of the input as required by keras.
            len_shape: Shape of positional features if used. Defaults to None.
        """
        inputs = keras.Input(shape=input_shape, name = "sentences")
        x = inputs

        if self.use_len_and_position:
            len_and_position = keras.Input(shape = len_shape, name="len_and_position")

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

        if self.use_len_and_position:
            x =  keras.layers.Concatenate()([x,len_and_position])

        x = Dense(20, activation = "relu")(x)
        outputs = Dense(5, activation="softmax")(x)

        if self.use_len_and_position:
            self.clf  = keras.Model(inputs = [inputs, len_and_position], outputs = outputs)
        else:
            self.clf = keras.Model(inputs, outputs)

        self.clf.summary()

    def train(self, X_train: np.array, y_train: np.array, X_val: np.array,
              y_val: np.array,abstractPosFeat_train:np.array = None, 
              abstractPosFeat_val:np.array=None, load_model: bool = False):

        if self.use_len_and_position:
            len_shape = (abstractPosFeat_train.shape[-1],)
            filename = f"{MODEL_CHECKPOINTS_PATH}/ResNet_1D_abstractPos_"
            
        else:
            len_shape = None
            filename = f"{MODEL_CHECKPOINTS_PATH}/ResNet_1D_"

        if load_model:
            print("Loading model...")
            self.clf = keras.models.load_model(
                filename + self.embedding_type)
            return

        print("Fitting model...")
        set_seeds()
        self.init_model(input_shape=(X_train.shape[-1]), len_shape=len_shape)

        callback = tf.keras.callbacks.EarlyStopping(patience=3)

        self.clf.compile("adam",
                         "sparse_categorical_crossentropy",
                         metrics=["accuracy"])

        if self.use_len_and_position:
            train_data = {"sentences": X_train, "len_and_position": abstractPosFeat_train}
            val_data = [X_val,abstractPosFeat_val]
        else:
            train_data = {"sentences": X_train}
            val_data = X_val

        self.clf.fit(train_data,
                     y_train,
                     batch_size=512,
                     epochs=10,
                     validation_data=(val_data, y_val),
                     callbacks=[callback])

        print("Saving model...")
        self.clf.save(filename +
                      self.embedding_type)

    def predict(self, X_test:np.array,abstractPosFeat_test:np.array = None ):
        if self.use_len_and_position:
            test_data = {"sentences": X_test, "len_and_position": abstractPosFeat_test}
        else:
            test_data = {"sentences": X_test}

        print("Making predictions...")
        return self.clf.predict(test_data)
