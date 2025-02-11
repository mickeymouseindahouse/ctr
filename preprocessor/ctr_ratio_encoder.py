from preprocessor.base_prepocessor import BasePreprocessor
from sklearn.base import TransformerMixin
from constants import TARGET_COLUMN

class CTREncoder(BasePreprocessor, TransformerMixin):
    """
    the preprocessor that uses mean ctr values per feature inplace of the feature itself
    """
    def __init__(self, target_column=TARGET_COLUMN, y=None):
        self.target_column = target_column
        self.feature_maps = {}
        self.y = y

    def fit(self, X, y=None):
        df = X.copy(deep=True)
        if y ==None:
            y = self.y
        df[self.target_column] = y
        feature_columns = [c for c in df.columns if c != self.target_column]
        for feature in feature_columns:
            self.feature_maps[feature] = df.groupby(feature)[self.target_column].mean().to_dict()
        return self

    def transform(self, X, y=None):
        df = X.copy()
        for feature, mapping in self.feature_maps.items():
            df[feature] = df[feature].map(mapping).fillna(0)
        return df