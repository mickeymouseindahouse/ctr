import pandas as pd
from preprocessor.base_prepocessor import BasePreprocessor
from sklearn.base import TransformerMixin
import numpy as np
from constants import TARGET_COLUMN

class Sampler(BasePreprocessor, TransformerMixin):
    """
    the preprocessor that keeps all positive value rows (or fraction if positive_value_frac is not 1) and samples negative value rows.
    """

    def __init__(self,
                 target_column=TARGET_COLUMN,
                 positive_value=1,
                 positive_frac=1.0,
                 negative_frac=1.0,
                 random_state=42):
        super().__init__()
        self.preprocessor_name = 'UnderSampler'
        self.target_column = target_column
        self.df = pd.DataFrame()
        self.positive_value = positive_value
        self.positive_frac = positive_frac
        self.negative_frac = negative_frac
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.array = None):
        self.df = X.copy(deep=True)
        if y is not None:
            self.df[self.target_column] = y
        return self


class UnderSampler(Sampler):
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.target_column is None:
            self.target_column = self.df.columns[-1]
        only_positive = self.df.loc[self.df[self.target_column] == self.positive_value].sample(
            frac=self.positive_frac,
            random_state=self.random_state)
        only_negative = self.df.loc[self.df[self.target_column] != self.positive_value].sample(
            n=int(only_positive.shape[0] * self.negative_frac),
            random_state=self.random_state, replace=True)
        transformed_df = pd.concat([only_positive, only_negative])
        transformed_df = transformed_df.sample(frac=1)
        return transformed_df

class OverSampler(Sampler):
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.target_column is None:
            self.target_column = self.df.columns[-1]
        only_negative = self.df.loc[self.df[self.target_column] != self.positive_value].sample(
            frac=self.negative_frac,
            random_state=self.random_state)
        only_positive = self.df.loc[self.df[self.target_column] == self.positive_value].sample(
            n=int(only_negative.shape[0] * self.positive_frac),
            random_state=self.random_state, replace=True)
        transformed_df = pd.concat([only_positive, only_negative])
        transformed_df = transformed_df.sample(frac=1)
        return transformed_df
