import enum
from typing import Optional, Callable

import pandas as pd

from constants import getroot
from data_loader.train_loader_session_splitter import TrainLoaderSessionSplitter
from preprocessor.base_prepocessor import BasePreprocessor


class FillAlgo(enum.Enum):
    MODE = "mode",
    MEAN = "mean",
    MIN = "min",
    MAX = "max",
    CUSTOM = "custom"

    def function(self, custom_function: Optional[Callable] = None) -> Callable:
        functions = {
            FillAlgo.MODE: lambda x: pd.Series.mode(x).iloc[0],
            FillAlgo.MEAN: pd.Series.mean,
            FillAlgo.MIN: pd.Series.min,
            FillAlgo.MAX: pd.Series.max,
            FillAlgo.CUSTOM: lambda x: pd.Series.mode(x).iloc[0],
        }
        return functions[self]




class FillNaPreprocessor(BasePreprocessor):
    def __init__(self, columns: Optional[list[str]] = None,
                 fill_algo: FillAlgo = FillAlgo.MODE,
                 custom_function: Optional[Callable] = None,
                 result_path: str = ''):
        super().__init__(result_path=result_path)
        self.columns = columns
        self.custom_function = custom_function
        self.fill_algo = fill_algo
        self.fill_method = self.fill_algo.function(self.custom_function)
        self.fill_values = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.columns is not None:
            self.fill_values = {col: self.fill_method(X[col]) for col in self.columns}
        else:
            self.fill_values = {col: self.fill_method(X[col]) for col in X.columns}
        return self

    #def transform(self, X: pd.DataFrame, y=None):
    #  if self.fill_values is None:
    #      raise ValueError("FillNaPreprocessor should have been fit before transform")
    # return X[self.columns].fillna(value=self.fill_values) if self.columns else X.fillna(value=self.fill_values)

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
            if self.fill_values is None:
                raise ValueError("Fit the preprocessor first!")
            
            # Fill NaNs without dropping other columns
            X_filled = X.copy()
            if self.columns:
                for col in self.columns:
                    X_filled[col] = X_filled[col].fillna(self.fill_values[col])
                    if X_filled[col].dtype == 'float64':  # Convert float categoricals to strings
                        X_filled[col] = X_filled[col].astype(str)
            else:
                for col in X_filled.columns:
                    X_filled[col] = X_filled[col].fillna(self.fill_values[col])
                    if X_filled[col].dtype == 'float64':  # Convert float categoricals to strings
                        X_filled[col] = X_filled[col].astype(str)
            return X_filled

if __name__ == "__main__":
    loader = TrainLoaderSessionSplitter(train_file=getroot() + "/data/train_dataset_full.csv")
    loader.load_data()
    proc = FillNaPreprocessor()
    print(proc.fit_transform(loader.train_data))
