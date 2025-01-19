import enum
from typing import Optional, Callable

import numpy as np
import pandas as pd

from preprocessor import Preprocessor


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
            FillAlgo.CUSTOM: custom_function,
        }
        return functions[self]




class FillNaPreprocessor(Preprocessor):
    def __init__(self, columns: Optional[list[str]] = None,
                 fill_algo: FillAlgo = FillAlgo.MODE,
                 custom_function: Optional[Callable] = None):
        super().__init__()
        self.columns = columns
        self.fill_method = fill_algo.function(custom_function)
        self.fill_values = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.columns is not None:
            self.fill_values = {col: self.fill_method(X[col]) for col in self.columns}
        else:
            self.fill_values = {col: self.fill_method(X[col]) for col in X.columns}
        return self

    def transform(self, X: pd.DataFrame, y=None):
        if self.fill_values is None:
            raise ValueError("FillNaPreprocessor should have been fit before transform")
        return X[self.columns].fillna(value=self.fill_values) if self.columns else X.fillna(value=self.fill_values)



if __name__ == "__main__":
    df = pd.DataFrame({"a": [1, np.nan, 1], "b": [4, np.nan, np.nan]})
    preprocessor = FillNaPreprocessor()
    preprocessor.fit(df)
    print(preprocessor.transform(df))