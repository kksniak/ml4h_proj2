from preprocessing import Preprocessing
from embeddings import Embeddings
from evaluation import evaluate
from models.vanilla_NN import Vanilla_NN

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

### Fit Models (more models to be added)

#### Vanilla
word2vec_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=True)
model = Vanilla_NN("word2vec", word2vec_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=True)
y_pred = model.predict(X_test)
evaluate("vanilla_nn_word2vec", y_pred, y_test, save_results=True)

######################## ADD WORD2vec models here


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

######################## ADD fasttext models here



## Tokeniser + trainable keras embedding Layer only

### Create Embedding
embedding_creator = Embeddings("kerasEmbed", sentences_train, sentences_val,
                               sentences_test)
X_train, X_val, X_test = embedding_creator.train()

### Fit Models (more models to be added)

#### Vanilla NN
keras_embedding_layers = embedding_creator.get_embedding_layer(
    mean_embedding=True)
model = Vanilla_NN("kerasEmbed", keras_embedding_layers)
model.train(X_train, y_train, X_val, y_val, load_model=False)
y_pred = model.predict(X_test)
evaluate("vanilla_nn_kerasEmbed", y_pred, y_test, save_results=True)

######################## ADD keras embedding models here
