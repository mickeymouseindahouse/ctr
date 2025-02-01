import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

from preprocessor.base_prepocessor import BasePreprocessor


class OverSampler(BasePreprocessor):
    def __init__(self, random_state=42, return_split=False):
        super().__init__()
        self.random_state = random_state
        self.return_split = return_split
        self.smote = SMOTE(sampling_strategy='auto', random_state=random_state)

    def fit(self, X, y=None) -> 'OverSampler':
        return self

    def transform(self, X, y=None):
        if y is None:
            y = X[X.columns[-1]]
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        if self.return_split:
            return X_resampled, y_resampled
        else:
            return pd.DataFrame([X_resampled, y_resampled])

class OverUnderSampler(OverSampler):
    def __init__(self, random_state=42):
        super().__init__(random_state=random_state)
        self.smote = SMOTEENN(sampling_strategy='auto', random_state=random_state)