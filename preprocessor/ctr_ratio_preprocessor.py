from typing import Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from preprocessor.base_prepocessor import BasePreprocessor
from constants import ONE_HOT_ENCODER, SCALER, CTR_LABEL_ENCODER
from preprocessor.ctr_ratio_encoder import CTREncoder
from constants import TARGET_COLUMN

class RatioBasedPreprocessor(BasePreprocessor):

    def __init__(self, numeric_features: Optional[list[str]] = None,
                 one_hot_features: Optional[list[str]] = None,
                 label_features: Optional[list[str]] = None,
                 result_path: str = '',
                 target_column: str = TARGET_COLUMN):
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
            transformers.append((CTR_LABEL_ENCODER, CTREncoder(target_column=self.target_column, y=y), self.label_features))
        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")
        self.preprocessor.fit(X)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.preprocessor is None:
            raise ValueError(f"{self.preprocessor_name} should be fit before transform")
        return self.preprocessor.transform(X)
