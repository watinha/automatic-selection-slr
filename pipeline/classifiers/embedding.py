import np, random, gensim

from keras import layers
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, train_test_split

from pipeline.classifiers import YearsSplit

class EmbeddingClassifier:
    def __init__ (self, seed, n_splits=3):
        self._seed = seed
        self._embedding_matrix = None
        self._n_splits = n_splits

    def get_classifier (self, X, y, word_index):
        print('===== MLP Keras =====')
        # generate embedding matrix
        self._embedding_matrix = self.get_embeddings(word_index)

        def create_model (filters=32, kernel_size=3, neurons=1):
            input_dim = X.shape[1]
            model = Sequential()
            model.add(layers.Embedding(input_dim=self._vocab_size,
                                       output_dim=self._embedding_dim,
                                       weights=[self._embedding_matrix],
                                       input_length=self._maxlen,
                                       trainable=True))
            #model.add(layers.LSTM(units=neurons))
            #model.add(layers.Flatten())
            model.add(layers.Conv1D(filters, kernel_size, activation='relu'))
            model.add(layers.GlobalMaxPooling1D())
            model.add(layers.Dense(neurons, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
            #model.summary()
            return model

        print('===== Keras hyperparameter optimization =====')
        model = KerasClassifier(build_fn=create_model, epochs=150, verbose=0)
        params = {
            'filters': [32, 64, 128],
            'kernel_size': [3, 5, 7],
            'neurons': [1, 10, 20, 30, 50]
        }
        cfl = GridSearchCV(model, params, cv=2, scoring='accuracy')
        cfl.fit(X, y)
        for param, value in cfl.best_params_.items():
            print("%s : %s" % (param, value))

        model = KerasClassifier(build_fn=create_model, epochs=150, verbose=0)
        model.set_params(**cfl.best_params_)
        return model


    def execute (self, dataset):
        X = dataset['features']
        y = dataset['categories']
        groups = dataset['years']
        random.seed(self._seed)
        kfold = YearsSplit(n_splits=self._n_splits, years=groups)
        model = self.get_classifier(X, y, dataset['word_index'])
        scores = cross_validate(model, X, y, cv=kfold, scoring=['f1_macro', 'precision_macro', 'recall_macro'])
        print("OUR APPROACH F-measure: %s on average and %s SD" %
                (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std()))
        print("OUR APPROACH Precision: %s on average and %s SD" %
                (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std()))
        print("OUR APPROACH Recall: %s on average and %s SD" %
                (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std()))

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)
        scores['probabilities'] = probabilities[:, 1]
        scores['y_test'] = y_test

        correct_exclusion_rate = []
        threasholds = []
        missed = []
        fscore_threashold = []
        exclusion_baseline = []
        missed_baseline = []
        for train_index, test_index in kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            y_score = model.predict_proba(X_train)[:, 1]
            precision, recall, threasholds2 = metrics.precision_recall_curve(
                    y_train, y_score)
            y_score = model.predict_proba(X_test)[:, 1]
            if (threasholds2[0] > 0.5):
                threasholds2 = [0.5]
            matrix = metrics.confusion_matrix(
                    y_test, [ 0 if i < threasholds2[0] else 1 for i in y_score ])
            correct_exclusion_rate.append(
                    matrix[0, 0] /
                    (matrix[0, 0] + matrix[1, 1] + matrix[0, 1] + matrix[1, 0]))
            missed.append(matrix[1, 0] / (matrix[1, 1] + matrix[1, 0]))
            threasholds.append(threasholds2[0])
            fscore_threashold.append(metrics.f1_score(
                y_test, [ 0 if i < threasholds2[0] else 1 for i in y_score ]))

            matrix = metrics.confusion_matrix(
                    y_test, [ 0 if i <  0.5 else 1 for i in y_score ])
            exclusion_baseline.append(
                    matrix[0, 0] /
                    (matrix[0, 0] + matrix[1, 1] + matrix[0, 1] + matrix[1, 0]))
            missed_baseline.append(matrix[1, 0] / (matrix[1, 1] + matrix[1, 0]))

        scores['exclusion_rate'] = correct_exclusion_rate
        scores['threasholds'] = threasholds
        scores['missed'] = missed
        scores['fscore_threashold'] = fscore_threashold
        scores['exclusion_baseline'] = exclusion_baseline
        scores['missed_baseline'] = missed_baseline

        dataset['%s_scores' % self.classifier_name] = scores
        return dataset


class MLPGloveEmbeddings (EmbeddingClassifier):
    def __init__ (self, seed=42, activation='relu', embedding_dim=200, maxlen=500, glove_file='glove.6B.200d.txt'):
        EmbeddingClassifier.__init__(self, seed)
        self.classifier_name = 'MLPKerasGLOVEEmbedding'
        self._activation = activation
        self._seed = seed
        self._glove_file = glove_file
        self._embedding_dim = embedding_dim
        self._maxlen = maxlen

    def get_embeddings (self, word_index):
        print('===== Glove News Embeddings loading from %s =====' % (self._glove_file))
        embedding_dim = self._embedding_dim
        self._vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
        self._embedding_matrix = np.zeros((self._vocab_size, embedding_dim))
        with open(self._glove_file) as f:
            for line in f:
                word, *vector = line.split()
                if word in word_index:
                    idx = word_index[word]
                    self._embedding_matrix[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]
        return self._embedding_matrix


class MLPSEEmbeddings (EmbeddingClassifier):
    def __init__ (self, seed=42, activation='relu', embedding_dim=200, maxlen=500, gensim_file='SO_vectors_200.bin'):
        EmbeddingClassifier.__init__(self, seed)
        self.classifier_name = 'MLPKerasSEEmbedding'
        self._activation = activation
        self._seed = seed
        self._embedding_dim = embedding_dim
        self._maxlen = maxlen
        self._embedding_matrix = None
        self._gensim_file = gensim_file
        self._se_embeddings = None


    def get_embeddings (self, word_index):
        print('===== SE Embeddings loading from %s =====' % (self._gensim_file))
        if self._se_embeddings == None:
            self._se_embeddings = gensim.models.KeyedVectors.load_word2vec_format(self._gensim_file, binary=True)
        embedding_dim = self._embedding_dim
        self._vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
        self._embedding_matrix = np.zeros((self._vocab_size, embedding_dim))
        not_found = []

        for word in word_index.keys():
            try:
                idx = word_index[word]
                self._embedding_matrix[idx] = np.array(
                    self._se_embeddings.get_vector(word), dtype=np.float32)[:embedding_dim]
            except:
                not_found.append(word)

        print('Not in embedding: %s...' % (not_found))
        return self._embedding_matrix
