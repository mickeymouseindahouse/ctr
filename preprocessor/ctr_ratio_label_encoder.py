from sklearn.preprocessing import LabelEncoder
from preprocessor.ctr_ratio_preprocessor import CTREncoder
from constants import TARGET_COLUMN

class CTRLabelEncoder(CTREncoder):
    """
    the preprocessor that uses mean ctr values to create labels for features
    """
    def __init__(self, target_column=TARGET_COLUMN, y=None):
        super().__init__(target_column=target_column, y=y)

    def label_encode(self, feature_mapping):
        self.original_to_encoded_mapping = {}  # To store the mapping between original labels and encoded labels
        for feature, ctr_dict in feature_mapping.items():
            le = LabelEncoder()
            keys = list(ctr_dict.keys())
            le.fit(keys)
            encoded_keys = le.transform(keys)
            self.original_to_encoded_mapping[feature] = dict(zip(keys, encoded_keys))
        return self.original_to_encoded_mapping

    def fit(self, X, y=None):
        df, feature_columns = self.preprocess(X, y)

        self.feature_maps = {}
        for feature in feature_columns:
            self.feature_maps[feature] = df.groupby(feature)[self.target_column].mean().to_dict()

        self.implemented_mapping = self.label_encode(self.feature_maps)
        return self