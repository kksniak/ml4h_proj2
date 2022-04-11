from typing import Literal, Tuple, List
from xmlrpc.client import boolean
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Lambda
from keras import backend
from keras.initializers import Constant
import keras
import numpy as np

from utils import set_seeds

SEED = 2137


class Embeddings:

    def __init__(self, embedding_type: Literal["TF-IDF", "word2vec", "fastText",
                                               "kerasEmbed"],
                 sentences_train: list, sentences_val: list,
                 sentences_test: list) -> None:
        self.embedding_type = embedding_type
        self.sentences_train, self.sentences_val, self.sentences_test = sentences_train, sentences_val, sentences_test

    def train(
            self,
            load_model: boolean = False) -> Tuple[np.array, np.array, np.array]:
        if self.embedding_type == "TF-IDF":
            self.tf_idf()

        elif self.embedding_type == "word2vec" or self.embedding_type == "fastText":
            self.load_model = load_model
            self.embed_dim = 100
            self.gensim_model()

        elif self.embedding_type == "kerasEmbed":
            self.embed_dim = 100
            self.keras_embedding_layer()

        else:
            print("Unknown embedding type")
            return

        return self.X_train, self.X_val, self.X_test

    def tf_idf(self) -> None:
        set_seeds()
        print('Training TF-IDF Embedding...')
        vectorizer = TfidfVectorizer(max_features=500)
        self.X_train = vectorizer.fit_transform(self.sentences_train).toarray()
        self.X_val = vectorizer.transform(self.sentences_val).toarray()
        self.X_test = vectorizer.transform(self.sentences_test).toarray()

    def gensim_model(self) -> None:
        self.X_train = [[word
                         for word in str(text).split()]
                        for text in self.sentences_train]
        self.X_val = [
            [word for word in str(text).split()] for text in self.sentences_val
        ]
        self.X_test = [
            [word for word in str(text).split()] for text in self.sentences_test
        ]

        if self.embedding_type == "word2vec":
            self.train_word2vec()
        elif self.embedding_type == "fastText":
            self.train_fastText()

        # create word to embed dictonary
        vocab = list(self.model.wv.key_to_index.keys())
        self.word2vec_dict = dict()
        for word in vocab:
            self.word2vec_dict[word] = self.model.wv.get_vector(word)

        self.tokenise_and_pad()

    def train_word2vec(self) -> None:
        set_seeds()
        if self.load_model:
            print('Loading word2vec Embedding...')
            self.model = Word2Vec.load(
                'code/embeddings_checkpoints/word2vec.model')
        else:
            print('Training word2vec Embedding...')
            self.model = Word2Vec(min_count=1,
                                  vector_size=self.embed_dim,
                                  seed=SEED)
            self.model.build_vocab(self.X_train)
            self.model.train(self.X_train,
                             total_examples=len(self.X_train),
                             epochs=20)
            self.model.save('code/embeddings_checkpoints/word2vec.model')

    def train_fastText(self) -> None:
        set_seeds()
        if self.load_model:
            print('Loading fastText Embedding...')
            self.model = FastText.load(
                'code/embeddings_checkpoints/fasttext.model')
        else:
            print('Training fastText Embedding...')
            self.model = FastText(vector_size=self.embed_dim, seed=SEED)
            self.model.build_vocab(self.X_train)
            self.model.train(self.X_train,
                             total_examples=len(self.X_train),
                             epochs=20)
            self.model.save('code/embeddings_checkpoints/fasttext.model')

    def keras_embedding_layer(self) -> None:
        print('Adding keras Embedding layers...')
        self.X_train = [[word
                         for word in str(text).split()]
                        for text in self.sentences_train]
        self.X_val = [
            [word for word in str(text).split()] for text in self.sentences_val
        ]
        self.X_test = [
            [word for word in str(text).split()] for text in self.sentences_test
        ]
        self.tokenise_and_pad()

    def tokenise_and_pad(self) -> None:
        # find longest sentence
        self.max_sentence_len = 0
        for sentence in self.X_train:
            if len(sentence) > self.max_sentence_len:
                self.max_sentence_len = len(sentence)

        # tokenise words to integers
        self.tokenise = Tokenizer(oov_token=0)
        self.tokenise.fit_on_texts(self.X_train + self.X_val + self.X_test)
        self.X_train = self.tokenise.texts_to_sequences(self.X_train)
        self.X_val = self.tokenise.texts_to_sequences(self.X_val)
        self.X_test = self.tokenise.texts_to_sequences(self.X_test)

        # pad training sets
        self.X_train = pad_sequences(self.X_train,
                                     maxlen=self.max_sentence_len,
                                     padding='post')
        self.X_val = pad_sequences(self.X_val,
                                   maxlen=self.max_sentence_len,
                                   padding='post')
        self.X_test = pad_sequences(self.X_test,
                                    maxlen=self.max_sentence_len,
                                    padding='post')

    def get_embedding_layer(self,
                            mean_embedding: boolean = True
                           ) -> List[keras.layers.Layer]:
        if self.embedding_type == "TF-IDF":
            print("No additional embedding layers for TF-IDF...")
            return

        vocab_size = len(self.tokenise.word_index) + 1

        if self.embedding_type == "word2vec" or self.embedding_type == "fastText":
            embedding_matrix = np.zeros(shape=(vocab_size, self.embed_dim))
            for word, i in self.tokenise.word_index.items():
                embedding_vector = self.word2vec_dict.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

            embedding_layers = [
                Embedding(input_dim=vocab_size,
                          output_dim=self.embed_dim,
                          input_length=self.max_sentence_len,
                          embeddings_initializer=Constant(embedding_matrix),
                          trainable=False)
            ]
            if mean_embedding:
                embedding_layers.append(
                    Lambda(lambda x: backend.mean(x, axis=1)))

        elif self.embedding_type == "kerasEmbed":
            embedding_layers = [
                Embedding(input_dim=vocab_size,
                          output_dim=self.embed_dim,
                          input_length=self.max_sentence_len)
            ]
            if mean_embedding:
                embedding_layers.append(
                    Lambda(lambda x: backend.mean(x, axis=1)))

        return embedding_layers
