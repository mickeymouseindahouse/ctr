from preprocessor.base_prepocessor import BasePreprocessor
from sklearn.base import TransformerMixin
from constants import TARGET_COLUMN

class CTREncoder(BasePreprocessor, TransformerMixin):
    """
    the preprocessor that uses mean ctr values per feature inplace of the feature itself
    """
    def __init__(self, target_column=TARGET_COLUMN, y=None):
        self.target_column = target_column
        self.y = y
        self.implemented_mapping = {}

    def preprocess(self, X, y=None, return_feature_columns=True):
        """combine X and y into a dataframe"""
        df = X.copy(deep=True)
        if self.target_column not in df.columns:
            if y is None:
                y = self.y
            df[self.target_column] = y
        if return_feature_columns:
            feature_columns = [c for c in df.columns if c != self.target_column]
            return df, feature_columns
        else:
            return df

    def get_mean_ctr_per_feature(self, df, feature_columns):
        self.feature_maps = {}
        for feature in feature_columns:
            self.feature_maps[feature] = df.groupby(feature)[self.target_column].mean().to_dict()
        return self.feature_maps


    def fit(self, X, y=None):
        df, feature_columns = self.preprocess(X, y)
        self.implemented_mapping = self.get_mean_ctr_per_feature(df, feature_columns)
        return self

    def transform(self, X, y=None):
        df = X.copy(deep=True)
        for feature, mapping in self.implemented_mapping.items():
            df[feature] = df[feature].map(mapping).fillna(0)
        return df