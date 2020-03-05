import np, gensim

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class EmbeddingsFeatureSelection:
    def __init__ (self, loader, k=10000, random_state=42,
                  vectorizer=TfidfVectorizer()):
        self._k = k
        self._random_state = 42
        self._vectorizer = vectorizer
        self._loader = loader

    def execute (self, dataset):
        print('===== Feature selection - Embeddings Clustering =====')
        texts = [ text_data['content'] for text_data in dataset ]
        self._vectorizer.fit(texts)

        if (len(self._vectorizer.vocabulary_) < self._k):
            print('Number of unique words is smaller than number of clusters (%d < %d)' %
                    (len(self._vectorizer.vocabulary_), self._k))
            return dataset

        self._embedding_matrix = self._loader.load(self._vectorizer)

        print('===== K-Means %d =====' % (self._k))
        model = KMeans(n_clusters=self._k, random_state=self._random_state)
        model.fit(self._embedding_matrix)

        print('===== replacing similar words by similarity =====')
        for text_data in dataset:
            tokens = word_tokenize(text_data['content'])
            new_tokens = []
            for token in tokens:
                try:
                    word_embedding = self._embedding_matrix[self._vectorizer.vocabulary_[token]]
                    word_cluster = model.predict(np.array([word_embedding]))[0]
                    new_tokens.append('token' + str(word_cluster))
                except:
                    #print('Key not found in index, removing word...')
                    pass

            text_data['content'] = ' '.join(new_tokens)

        return dataset


class GloveEmbeddingLoader():
    def __init__ (self, glove_file='glove.6B.200d.txt', embedding_dim=200):
        self._glove_file = glove_file
        self._embedding_dim = embedding_dim

    def load (self, vectorizer):
        print('===== Glove Embeddings loading from %s =====' % (self._glove_file))
        embedding_dim = self._embedding_dim
        self._vocab_size = len(vectorizer.vocabulary_) + 1
        self._embedding_matrix = np.zeros((self._vocab_size, embedding_dim))
        with open(self._glove_file) as f:
            for line in f:
                word, *vector = line.split()
                if word in vectorizer.vocabulary_:
                    idx = vectorizer.vocabulary_[word]
                    self._embedding_matrix[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]
        return self._embedding_matrix


class GensimEmbeddingLoader():
    def __init__ (self, gensim_file='SO_vectors_200.bin', embedding_dim=200):
        self._gensim_file = gensim_file
        self._embedding_dim = embedding_dim

    def load (self, vectorizer):
        print('===== SE Embeddings loading from %s =====' % (self._gensim_file))
        self._se_embeddings = gensim.models.KeyedVectors.load_word2vec_format(self._gensim_file, binary=True)
        embedding_dim = self._embedding_dim
        self._vocab_size = len(vectorizer.vocabulary_) + 1  # Adding again 1 because of reserved 0 index
        self._embedding_matrix = np.zeros((self._vocab_size, embedding_dim))
        not_found = []

        for word in vectorizer.vocabulary_.keys():
            try:
                idx = vectorizer.vocabulary_[word]
                self._embedding_matrix[idx] = np.array(
                    self._se_embeddings.get_vector(word), dtype=np.float32)[:embedding_dim]
            except:
                not_found.append(word)

        print('Not in embedding: %s...' % (not_found))
        return self._embedding_matrix
