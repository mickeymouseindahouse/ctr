from typing import Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from preprocessor.base_prepocessor import BasePreprocessor
from constants import ONE_HOT_ENCODER, SCALER, CTR_LABEL_ENCODER
from sklearn.base import BaseEstimator, TransformerMixin


class CTRTransformer(BaseEstimator, TransformerMixin):
    """
    the preprocessor that uses mean ctr values per feature inplace of the feature itself
    """
    def __init__(self, target_column='is_click', y=None):
        self.target_column = target_column
        self.feature_maps = {}
        self.y = y

    def fit(self, X, y=None):
        df = X.copy()
        if y ==None:
            y = self.y
        df[self.target_column] = y
        feature_columns = [c for c in df.columns if c != self.target_column]
        for feature in feature_columns:
            self.feature_maps[feature] = df.groupby(feature)[self.target_column].mean().to_dict()
        return self

    def transform(self, X):
        df = X.copy()
        for feature, mapping in self.feature_maps.items():
            df[feature] = df[feature].map(mapping).fillna(0)
        return df


class RatioBasedPreprocessor(BasePreprocessor):

    def __init__(self, numeric_features: Optional[list[str]] = None,
                 one_hot_features: Optional[list[str]] = None,
                 label_features: Optional[list[str]] = None,
                 result_path: str = '',
                 target_column: str = 'is_click'):
        self.result_path = result_path
        super().__init__(result_path=result_path)
        self.preprocessor = None
        self.numeric_features = numeric_features
        self.one_hot_features = one_hot_features
        self.label_features = label_features
        self.preprocessor_name = 'RatioBasedPreprocessor'
        self.target_column = target_column

    def fit(self, X: pd.DataFrame, y=None):
        transformers = []
        if self.numeric_features:
            transformers.append((SCALER, StandardScaler(), self.numeric_features))
        if self.one_hot_features:
            transformers.append((ONE_HOT_ENCODER, OneHotEncoder(drop='first'), self.one_hot_features))
        if self.label_features:
            transformers.append((CTR_LABEL_ENCODER, CTRTransformer(target_column=self.target_column, y=y), self.label_features))
        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")
        self.preprocessor.fit(X)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.preprocessor is None:
            raise ValueError(f"{self.preprocessor_name} should be fit before transform")
        return self.preprocessor.transform(X)
