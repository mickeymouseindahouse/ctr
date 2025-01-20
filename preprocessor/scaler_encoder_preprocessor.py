from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder

from preprocessor.base_prepocessor import BasePreprocessor
from constants import ONE_HOT_ENCODER, LABEL_ENCODER, SCALER


class ScalerEncoderPreprocessor(BasePreprocessor):
    def __init__(self, numeric_features: Optional[list[str]]=None, one_hot_features: Optional[list[str]]=None,
                 label_features: Optional[list[str]]=None):
        super().__init__()
        self.preprocessor = None
        self.numeric_features = numeric_features
        self.one_hot_features = one_hot_features
        self.label_features = label_features

    def fit(self, X: pd.DataFrame, y=None):
        self.preprocessor = ColumnTransformer(transformers=[], remainder="passthrough")
        if self.numeric_features:
            self.preprocessor.transformers.append((SCALER, StandardScaler(), self.numeric_features))
        if self.one_hot_features:
            self.preprocessor.transformers.append((ONE_HOT_ENCODER, OneHotEncoder(), self.one_hot_features))
        if self.label_features:
            self.preprocessor.transformers.append((LABEL_ENCODER, OrdinalEncoder(), self.label_features))
        self.preprocessor.fit(X)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.preprocessor is None:
            raise ValueError("ScalerEncoderPreprocessor should be fit before transform")
        return self.preprocessor.transform(X)
