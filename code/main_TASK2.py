from preprocessing import Preprocessing
from embeddings import Embeddings
from evaluation import evaluate
from models.vanilla_NN import Vanilla_NN
from models.bidirectional_LSTM_POS import BiRNN_LSTM_POS
from models.bidirectional_LSTM import BiRNN_LSTM
from models.resnet1d import ResNet1D_model
from utils import get_POS_encoding

import numpy as np

SEED = 2137

#############
#############
# TASK 2
#############
#############

### Loading and Preprocessing Datasets
dataset_preprocessing = Preprocessing(stemming=False, lemmatisation=True)
dataset_preprocessing.preprocess_datasets()
sentences_train, sentences_val, sentences_test, y_train, y_val, y_test, abstractPosFeat_train, abstractPosFeat_val, abstractPosFeat_test = dataset_preprocessing.get_X_and_encoded_Y(
)

## Word2Vec

### Create Embedding

embedding_creator = Embeddings("word2vec", sentences_train, sentences_val,
                               sentences_test)
X_train, X_val, X_test = embedding_creator.train(load_model=True)

### Fit Models

#### word2vec with mean embedding
word2vec_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=True)

#### Vanilla
model = Vanilla_NN("word2vec", word2vec_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=True)
y_pred = model.predict(X_test)
evaluate("vanilla_nn_word2vec", y_pred, y_test, save_results=True)


#### word2vec with concatenated embedding
word2vec_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=False)

#### BiLSTM 
model = BiRNN_LSTM("word2vec", word2vec_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=True)
y_pred = model.predict(X_test)
evaluate("BiLSTM_word2vec", y_pred, y_test, save_results=True)


pos, pos_val, pos_test = get_POS_encoding(sentences_train,sentences_val,sentences_test,True)


model = BiRNN_LSTM_POS("word2vec", word2vec_embedding_layers,use_len_and_position= True)
model.train(X_train, y_train, X_val, y_val,pos_train=pos, pos_val=pos_val,
            abstractPosFeat_train=abstractPosFeat_train,abstractPosFeat_val=abstractPosFeat_val, load_model=True)
y_pred = model.predict(X_test, pos_test,abstractPosFeat_test)
evaluate("BiLSTM_POS_word2vec", y_pred, y_test, save_results=True)

#### ResNet1D with position
model =ResNet1D_model("word2vec",word2vec_embedding_layers,use_len_and_position = True)
model.train(X_train,y_train, X_val, y_val,abstractPosFeat_train=abstractPosFeat_train,
            abstractPosFeat_val=abstractPosFeat_val, load_model=False)
y_pred = model.predict(X_test,abstractPosFeat_test)
evaluate("resnet1d_position_word2vec", y_pred, y_test, save_results=True)



## FastText

### Create Embedding
embedding_creator = Embeddings("fastText", sentences_train, sentences_val,
                               sentences_test)
X_train, X_val, X_test = embedding_creator.train(load_model=True)

### Fit Models

#### FastText with mean embedding
fasttext_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=True)

#### Vanilla NN
model = Vanilla_NN("fastText", fasttext_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=True)
y_pred = model.predict(X_test)
evaluate("vanilla_nn_fastText", y_pred, y_test, save_results=True)

#### FastText with concatenated embedding
fasttext_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=False)

#### BiLSTM 
model = BiRNN_LSTM("fastText", fasttext_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=True)
y_pred = model.predict(X_test)
evaluate("BiLSTM_fastText", y_pred, y_test, save_results=True)

#### BiLSTM with POS and position
model = BiRNN_LSTM_POS("fastText", fasttext_embedding_layers,use_len_and_position= True)
model.train(X_train, y_train, X_val, y_val,pos_train=pos, pos_val=pos_val,
            abstractPosFeat_train=abstractPosFeat_train,abstractPosFeat_val=abstractPosFeat_val, load_model=False)
y_pred = model.predict(X_test, pos_test,abstractPosFeat_test)
evaluate("BiLSTM_POS_fastText", y_pred, y_test, save_results=True)

#### ResNet1D with position
model =ResNet1D_model("fastText",fasttext_embedding_layers,use_len_and_position = True)
model.train(X_train,y_train, X_val, y_val,abstractPosFeat_train=abstractPosFeat_train,
            abstractPosFeat_val=abstractPosFeat_val, load_model=True)
y_pred = model.predict(X_test,abstractPosFeat_test)
evaluate("resnet1d_position_fastText", y_pred, y_test, save_results=True)


## Trainable keras embedding Layer

### Create Embedding
embedding_creator = Embeddings("kerasEmbed", sentences_train, sentences_val,
                               sentences_test)
X_train, X_val, X_test = embedding_creator.train()

### Fit Models

#### Keras Embedding Layer with mean embedding
keras_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=True)

#### Vanilla NN
model = Vanilla_NN("kerasEmbed", keras_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=True)
y_pred = model.predict(X_test)
evaluate("vanilla_nn_kerasEmbed", y_pred, y_test, save_results=True)

#### Keras Layer embedding with concatenated embedding
keras_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=False)

#### BiLSTM 
model = BiRNN_LSTM("kerasEmbed", keras_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=True)
y_pred = model.predict(X_test)
evaluate("BiLSTM_kerasEmbed", y_pred, y_test, save_results=True)

#### BiLSTM with POS and position
model = BiRNN_LSTM_POS("kerasEmbed", keras_embedding_layers,use_len_and_position= True)
model.train(X_train, y_train, X_val, y_val,pos_train=pos, pos_val=pos_val,
            abstractPosFeat_train=abstractPosFeat_train,abstractPosFeat_val=abstractPosFeat_val, load_model=True)
y_pred = model.predict(X_test, pos_test,abstractPosFeat_test)
evaluate("BiLSTM_POS_kerasEmbed", y_pred, y_test, save_results=True)

#### ResNet1D with position
model =ResNet1D_model("kerasEmbed",keras_embedding_layers,use_len_and_position = True)
model.train(X_train,y_train, X_val, y_val,abstractPosFeat_train=abstractPosFeat_train,
            abstractPosFeat_val=abstractPosFeat_val, load_model=True)
y_pred = model.predict(X_test,abstractPosFeat_test)
evaluate("resnet1d_position_kerasEmbed", y_pred, y_test, save_results=True)

