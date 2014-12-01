
from sklearn.base import BaseEstimator, TransformerMixin # hack for PCA

class DenseMatrixTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.todense()
