from embeddings import Embeddings

if __name__ == '__main__':

    similar_words = ['doctor', 'pain', 'medicine']

    embedding_creator = Embeddings("word2vec", [], [], [])
    _, _, _ = embedding_creator.train(load_model=True)
    print("\n\nThe word2vec embedding analysis:")
    for word in similar_words:
        print("\n -> Similar to '" + word + "':")
        print(embedding_creator.model.wv.most_similar(positive=[word], negative=[], topn=5))

    print("\n -> Similar to 'cardiologist' - 'heart' + 'skin' =")
    print(embedding_creator.model.wv.most_similar(positive=['cardiologist', 'skin'], negative=['heart'], topn=3))
    print("\n -> Similar to 'skin' - 'dermatologist' + 'ophthalmologist' =")
    print(embedding_creator.model.wv.most_similar(positive=['skin','ophthalmologist'], negative=['dermatologist'], topn=3))

    print("\n\nThe fastText analysis:")
    embedding_creator = Embeddings("fastText", [], [], [])
    _, _, _ = embedding_creator.train(load_model=True)

    for word in similar_words:
        print("\n -> Similar to '" + word + "':")
        print(embedding_creator.model.wv.most_similar(positive=[word], negative=[], topn=5))

    print("\n -> Similar to 'cardiologist' - 'heart' + 'skin' =")
    print(embedding_creator.model.wv.most_similar(positive=['cardiologist', 'skin'], negative=['heart'], topn=3))
    print("\n -> Similar to 'skin' - 'dermatologist' + 'ophthalmologist' =")
    print(embedding_creator.model.wv.most_similar(positive=['skin','ophthalmologist'], negative=['dermatologist'], topn=3))



