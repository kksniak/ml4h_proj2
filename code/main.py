from preprocessing import Preprocessing
from embeddings import Embeddings
from evaluation import evaluate
from models.vanilla_NN import Vanilla_NN
from models.conv_1D import Conv1D_model
from models.bidirectional_LSTM import BiRNN_LSTM

#############
#############
# TASK 1
#############
#############

### Loading and Preprocessing Datasets
dataset_preprocessing = Preprocessing(stemming=False, lemmatisation=False)
dataset_preprocessing.preprocess_datasets()
sentences_train, sentences_val, sentences_test, y_train, y_val, y_test = dataset_preprocessing.get_X_and_encoded_Y(
)

### Create TF-IDF Embedding
embedding_creator = Embeddings("TF-IDF", sentences_train, sentences_val,
                               sentences_test)
X_train, X_val, X_test = embedding_creator.train()

### Fit Models (more models to be added...)

#### Vanilla NN
model = Vanilla_NN("TF-IDF")
model.train(X_train, y_train, X_val, y_val, load_model=True)
y_pred = model.predict(X_test)
evaluate("vanilla_nn_tf_idf", y_pred, y_test, save_results=True)


#############
#############
# TASK 2
#############
#############

## Word2Vec

### Create Embedding

embedding_creator = Embeddings("word2vec", sentences_train, sentences_val,
                               sentences_test)
X_train, X_val, X_test = embedding_creator.train(load_model=True)

### Fit Models (more models to be added)

#### Vanilla
word2vec_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=True)
model = Vanilla_NN("word2vec", word2vec_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=True)
y_pred = model.predict(X_test)
evaluate("vanilla_nn_word2vec", y_pred, y_test, save_results=True)

#### Conv 1D
word2vec_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=False)
model = Conv1D_model("word2vec", word2vec_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=True)
y_pred = model.predict(X_test)
evaluate("conv1d_word2vec", y_pred, y_test, save_results=True)

#### Bidirectional RNN + LSTM
word2vec_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=False)
model = BiRNN_LSTM("word2vec", word2vec_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=False)
y_pred = model.predict(X_test)
evaluate("birnn_word2vec", y_pred, y_test, save_results=True)

## FastText

### Create Embedding
embedding_creator = Embeddings("fastText", sentences_train, sentences_val,
                               sentences_test)
X_train, X_val, X_test = embedding_creator.train(load_model=True)

### Fit Models (more models to be added)

#### Vanilla NN
fasttext_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=True)
model = Vanilla_NN("fastText", fasttext_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=True)
y_pred = model.predict(X_test)
evaluate("vanilla_nn_fastText", y_pred, y_test, save_results=True)

#### Conv 1D
fasttext_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=False)
model = Conv1D_model("fastText", fasttext_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=True)
y_pred = model.predict(X_test)
evaluate("conv1d_fasttext", y_pred, y_test, save_results=True)

#### Bidirectional RNN + LSTM
fasttext_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=False)
model = BiRNN_LSTM("fastText", fasttext_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=False)
y_pred = model.predict(X_test)
evaluate("BiRNN_fasttext", y_pred, y_test, save_results=True)

## Tokeniser + trainable keras embedding Layer only

### Create Embedding
embedding_creator = Embeddings("kerasEmbed", sentences_train, sentences_val,
                               sentences_test)
X_train, X_val, X_test = embedding_creator.train(load_model=True)

### Fit Models (more models to be added)

#### Vanilla NN
keras_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=True)
model = Vanilla_NN("kerasEmbed", keras_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=False)
y_pred = model.predict(X_test)
evaluate("vanilla_nn_kerasEmbed", y_pred, y_test, save_results=False)

#### Conv 1D
fasttext_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=False)
model = Conv1D_model("kerasEmbed", keras_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=False)
y_pred = model.predict(X_test)
evaluate("conv1d_keras_embed", y_pred, y_test, save_results=True)

#### Bidirectional RNN + LSTM
fasttext_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=False)
model = BiRNN_LSTM("kerasEmbed", keras_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=False)
y_pred = model.predict(X_test)
evaluate("BiRNN_kerasembed", y_pred, y_test, save_results=True)

#############
#############
# TASK 3
#############
#############