import pandas as pd
from sklearn.decomposition import PCA

from preprocessor.base_prepocessor import BasePreprocessor


class PcaPreprocessor(BasePreprocessor):
    def __init__(self, n_components=None, random_state=42):
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state
        self.pca = None

    def fit(self, X: pd.DataFrame, y=None) -> 'BasePreprocessor':
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.pca is None:
            raise "PCA preprocessor not fit!"
        return pd.DataFrame(self.pca.transform(X))