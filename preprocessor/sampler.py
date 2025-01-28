import pandas as pd
from preprocessor.base_prepocessor import BasePreprocessor
from sklearn.base import TransformerMixin
import numpy as np


class Sampler(BasePreprocessor, TransformerMixin):
    """
    the preprocessor that keeps all positive value rows (or fraction if positive_value_frac is not 1) and samples negative value rows.
    """

    def __init__(self,
                 sample_number=None, # if keeping as None, should take the same amount as positive values
                 return_splited=False, # return as X,y if True
                 target_column='is_click',
                 positive_value=1,
                 negative_value=None, # not the positive
                 positive_value_frac=1,
                 random_state=42):
        super().__init__()
        self.preprocessor_name = 'UnderSampler'
        self.target_column = target_column
        self.sample_number = sample_number
        self.return_splited = return_splited
        self.df = pd.DataFrame()
        self.positive_value = positive_value
        self.positive_value_frac = positive_value_frac
        self.negative_value = negative_value
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.array = None):
        self.df = X.copy(deep=True)
        if y is not None:
            self.df[self.target_column] = y
        return self

    def transform(self, X: pd.DataFrame = None, y=None):
        if self.target_column is None:
            self.target_column = self.df.columns[-1]
        self.df[self.target_column] = np.where(self.df[self.target_column], self.df[self.target_column]>0, -1)
        only_positive = self.df.loc[self.df[self.target_column] == self.positive_value].sample(frac=self.positive_value_frac,
                                                                                        random_state=self.random_state)
        if self.negative_value is None:
           self.negative_value = [value for value in self.df[self.target_column].unique() if value != self.positive_value][0]
        if self.sample_number is not None:
            only_negative = self.df.loc[self.df[self.target_column] == self.negative_value]
            if isinstance(self.sample_number, int):
                only_negative = only_negative.sample(frac=1, random_state=self.random_state).head(self.sample_number)
            elif isinstance(self.sample_number, float):
                only_negative = only_negative.sample(frac=self.sample_number, random_state=self.random_state)
        else:
            available_0s = self.df.loc[self.df[self.target_column] == self.negative_value]
            available_length = min(len(only_positive), available_0s.shape[0])
            only_negative = pd.DataFrame(available_0s.sample(frac=1, random_state=self.random_state).head(available_length))
        transformed_df = pd.concat([only_positive, only_negative])
        transformed_df = transformed_df.sample(frac=1)
        if self.return_splited:
            return transformed_df[[column for column in transformed_df.columns if column != self.target_column]], \
            transformed_df[self.target_column]
        else:
            return transformed_df
