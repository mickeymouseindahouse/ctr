from typing import Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from preprocessor.base_prepocessor import BasePreprocessor
from constants import ONE_HOT_ENCODER, SCALER, CTR_LABEL_ENCODER
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from preprocessor.ctr_ratio_preprocessor import RatioBasedPreprocessor, CTREncoder

if False:
  class CTRLabelEncoder(CTREncoder):
      """
      the preprocessor that uses mean ctr values to create labels for features
      """
      def __init__(self, target_column='is_click', y=None):
          super().__init__()
          self.target_column = target_column
          self.feature_maps = {}
          self.original_to_encoded_mapping = {}
          self.y = y

      def fit(self, X, y=None):
          df = X.copy()
          if y is None:
              y = self.y
          df[self.target_column] = y
          feature_columns = [c for c in df.columns if c != self.target_column]
          for feature in feature_columns:
              self.feature_maps[feature] = df.groupby(feature)[self.target_column].mean().to_dict()
          self.original_to_encoded_mapping = {}  # To store the mapping between original labels and encoded labels
          for feature, ctr_dict in self.feature_maps.items():
              le = LabelEncoder()
              keys = list(ctr_dict.keys())
              le.fit(keys)
              encoded_keys = le.transform(keys)
              self.original_to_encoded_mapping[feature] = dict(zip(keys, encoded_keys))
          return self

      def transform(self, X, y=None):
          df = X.copy()
          for feature, mapping in self.original_to_encoded_mapping.items():
              df[feature] = df[feature].map(mapping).fillna(0)
          return df

class CTRLabelEncoder(BasePreprocessor):
    def __init__(self, target_column='is_click', y=None):
      super().__init__()
      self.target_column = target_column
      self.feature_maps = {}
      self.original_to_encoded_mapping = {}
      self.y = y
      
    def fit(self, X: pd.DataFrame, y=None):
        # No need for manual encoding - CatBoost handles it
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X  # Just pass through


class RatioBasedLabelPreprocessor(RatioBasedPreprocessor):
    def __init__(
        self,
        numeric_features: Optional[list[str]] = None,
        label_features: Optional[list[str]] = None, 
        result_path: str = '',
        target_column: str = 'is_click'
    ):
        super().__init__(result_path=result_path)
        self.numeric_features = numeric_features
        self.label_features = label_features
        self.target_column = target_column
        self.scaler = None
        self.label_encoder = None  

    def fit(self, X: pd.DataFrame, y=None):
        if self.numeric_features:
            self.scaler = StandardScaler().fit(X[self.numeric_features])
        if self.label_features:
            self.label_encoder = CTRLabelEncoder(target_column=self.target_column, y=y).fit(X[self.label_features], y)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_transformed = X.copy()
        if self.numeric_features and self.scaler:
            X_transformed[self.numeric_features] = self.scaler.transform(X[self.numeric_features])
        if self.label_features and self.label_encoder:
            X_transformed[self.label_features] = self.label_encoder.transform(X[self.label_features])
        return X_transformed
