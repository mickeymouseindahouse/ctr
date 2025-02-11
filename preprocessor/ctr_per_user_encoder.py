from sklearn.preprocessing import LabelEncoder
from preprocessor.ctr_ratio_preprocessor import CTREncoder
from constants import TARGET_COLUMN

class CTRPerUserEncoder(CTREncoder):
    """
    the preprocessor uses CTR as a separate feature per user_id
    """
    def __init__(self, target_column=TARGET_COLUMN, y=None):
        super().__init__(target_column=target_column, y=y)
        self.feature_maps = {}
        self.original_to_encoded_mapping = {}

    def fit(self, X, y=None):
        df = X.copy(deep=True)
        if y ==None:
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