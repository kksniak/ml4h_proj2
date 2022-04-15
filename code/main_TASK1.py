from preprocessing import Preprocessing
from embeddings import Embeddings
from evaluation import evaluate
from models.vanilla_NN import Vanilla_NN
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB

from utils import fast_feature_selector

SEED = 2137

#############
#############
# TASK 1
#############
#############

## WITHOUT stemming, WITHOUT lemmatisation

### Loading and Preprocessing Datasets
dataset_preprocessing = Preprocessing(stemming=False, lemmatisation=False)
dataset_preprocessing.preprocess_datasets()
sentences_train, sentences_val, sentences_test, y_train, y_val, y_test, _, _, _ = dataset_preprocessing.get_X_and_encoded_Y(
)

### Create TF-IDF Embedding
embedding_creator = Embeddings("TF-IDF", sentences_train, sentences_val,
                               sentences_test)
X_train, X_val, X_test = embedding_creator.train()

### Fit Models

#### Vanilla NN
model = Vanilla_NN("TF-IDF")
model.train(X_train, y_train, X_val, y_val, load_model=True, save_name = 'no_lem_no_stem')
y_pred = model.predict(X_test)
evaluate("vanilla_nn_tf_idf_no_lem_no_stem", y_pred, y_test, save_results=True)

#### GaussianNB
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)
evaluate("MultinomialNB_tf_idf_no_lem_no_stem", y_pred, y_test, save_results=True)

###### feature selection for simple models
X_train, X_test = fast_feature_selector(200, X_train, y_train, X_test)

#### LGBM
model = LGBMClassifier(n_estimators = 150, seed = SEED)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)
evaluate("lgbm_tf_idf_no_lem_no_stem", y_pred, y_test, save_results=True)

## WITH stemming, WITHOUT lemmatisation

### Loading and Preprocessing Datasets
dataset_preprocessing = Preprocessing(stemming=True, lemmatisation=False)
dataset_preprocessing.preprocess_datasets()
sentences_train, sentences_val, sentences_test, y_train, y_val, y_test, _, _, _ = dataset_preprocessing.get_X_and_encoded_Y(
)

### Create TF-IDF Embedding
embedding_creator = Embeddings("TF-IDF", sentences_train, sentences_val,
                               sentences_test)
X_train, X_val, X_test = embedding_creator.train()

### Fit Models

#### Vanilla NN
model = Vanilla_NN("TF-IDF") 
model.train(X_train, y_train, X_val, y_val, load_model=True, save_name = 'stem')
y_pred = model.predict(X_test)
evaluate("vanilla_nn_tf_idf_with_lem_no_stem", y_pred, y_test, save_results=True)

#### GaussianNB
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)
evaluate("MultinomialNB_tf_idf_with_lem_no_stem", y_pred, y_test, save_results=True)

###### feature selection for simple models
X_train, X_test = fast_feature_selector(200, X_train, y_train, X_test)

#### LGBM
model = LGBMClassifier(n_estimators = 150, seed = SEED)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)
evaluate("lgbm_tf_idf_with_lem_no_stem", y_pred, y_test, save_results=True)

## WITHOUT stemming, WITH lemmatisation

### Loading and Preprocessing Datasets
dataset_preprocessing = Preprocessing(stemming=False, lemmatisation=True)
dataset_preprocessing.preprocess_datasets()
sentences_train, sentences_val, sentences_test, y_train, y_val, y_test, _, _, _ = dataset_preprocessing.get_X_and_encoded_Y(
)

### Create TF-IDF Embedding
embedding_creator = Embeddings("TF-IDF", sentences_train, sentences_val,
                               sentences_test)
X_train, X_val, X_test = embedding_creator.train()

### Fit Models

#### Vanilla NN
model = Vanilla_NN("TF-IDF")
model.train(X_train, y_train, X_val, y_val, load_model=True, save_name = 'lem')
y_pred = model.predict(X_test)
evaluate("vanilla_nn_tf_idf_no_lem_with_stem", y_pred, y_test, save_results=True)

#### GaussianNB
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)
evaluate("MultinomialNB_tf_idf_no_lem_with_stem", y_pred, y_test, save_results=True)

###### feature selection for simple models
X_train, X_test = fast_feature_selector(200, X_train, y_train, X_test)

#### LGBM
model = LGBMClassifier(n_estimators = 150, seed = SEED)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)
evaluate("lgbm_tf_idf_no_lem_with_stem", y_pred, y_test, save_results=True)