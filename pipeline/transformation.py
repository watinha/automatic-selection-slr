import np

from sklearn.decomposition import TruncatedSVD

class LSATransformation:
    def __init__ (self, n_components = 100, random_state = 42):
        self._ncomponents = n_components
        self._lsa = TruncatedSVD(n_components = n_components, random_state=random_state)

    def execute (self, dataset):
        print('===== LSA Transformation =====')
        dataset['features'] = np.array(self._lsa.fit_transform(dataset['features']))
        print(dataset['features'].shape)
        return dataset
