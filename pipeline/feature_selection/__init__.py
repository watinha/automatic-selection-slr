from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, VarianceThreshold, SelectKBest, chi2

class RFECVFeatureSelection:
    def __init__ (self, estimator):
        self._rfecv = RFECV(estimator = estimator,
                cv=StratifiedKFold(5), scoring='recall')

    def execute (self, dataset):
        print('===== Feature selection - RFECV =====')
        dataset['features'] = self._rfecv.fit_transform(dataset['features'].toarray(), dataset['categories'])
        print(dataset['features'].shape)
        return dataset


class VarianceThresholdFeatureSelection:
    def __init__ (self, threshold):
        self._variance_threshold = VarianceThreshold(threshold=threshold)

    def execute (self, dataset):
        print('===== Feature selection - Variance Threshold =====')
        dataset['features'] = self._variance_threshold.fit_transform(dataset['features'])
        print(dataset['features'].shape)
        return dataset


class SelectKBestSelection:
    def __init__ (self, k):
        self._k = k
        self._select_k_best = SelectKBest(chi2, k=k)

    def execute (self, dataset):
        print('===== Feature selection - SelectKBest =====')

        if dataset['features'].shape[1] < self._k:
            return dataset

        dataset['features'] = self._select_k_best.fit_transform(dataset['features'], dataset['categories'])
        print(dataset['features'].shape)
        return dataset


class USESFeatureSelection:
    def __init__ (self, k=100):
        self._k = k
        self._affinity_score = []

    def _affinity (self, word_frequency_column, categories, category):
        ncw = 0
        nc = 0
        nw = 0
        for i in range(0, len(categories)):
            if categories[i] == category: nc += 1
            if word_frequency_column[i] > 0: nw += 1
            if categories[i] == category and word_frequency_column[i] > 0: ncw += 1
        return ncw / (nc + nw - ncw)


    def _score (self, features, categories):
        n_words = features.shape[1]
        self._affinity_score = [
            self._affinity(features[:,i], categories, 1) -
            self._affinity(features[:,i], categories, 0)
            for i in range(0, n_words) ]
        return (self._affinity_score, [])


    def execute (self, dataset):
        print('===== Feature selection - USES =====')
        X = dataset['features']
        y = dataset['categories']
        fs = SelectKBest(self._score, k=self._k)
        dataset['features'] = fs.fit_transform(X, y)
        print(dataset['features'].shape)
        return dataset
