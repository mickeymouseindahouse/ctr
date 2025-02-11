from typing import Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from constants import ONE_HOT_ENCODER, SCALER, CTR_LABEL_ENCODER, TARGET_COLUMN
from preprocessor.ctr_ratio_preprocessor import RatioBasedPreprocessor
from preprocessor.ctr_ratio_label_encoder import CTRLabelEncoder
from constants import TARGET_COLUMN

class RatioBasedLabelPreprocessor(RatioBasedPreprocessor):

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
        self.preprocessor_name = 'RatioBasedLabelPreprocessor'
        self.target_column = target_column

    def fit(self, X: pd.DataFrame, y=None):
        transformers = []
        if self.numeric_features:
            transformers.append((SCALER, StandardScaler(), self.numeric_features))
        if self.one_hot_features:
            transformers.append((ONE_HOT_ENCODER, OneHotEncoder(drop='first'), self.one_hot_features))
        if self.label_features:
            transformers.append((CTR_LABEL_ENCODER, CTRLabelEncoder(target_column=self.target_column, y=y), self.label_features))
        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")
        self.preprocessor.fit(X)
        return self

