from preprocessor.ctr_ratio_preprocessor import CTREncoder
from constants import TARGET_COLUMN

class CTRPerUserEncoder(CTREncoder):
    """
    the preprocessor uses CTR as a separate feature per user_id
    """
    def __init__(self, target_column=TARGET_COLUMN, y=None, group_on_column = 'user_id'):
        super().__init__(target_column=target_column, y=y)
        self.group_on_column = group_on_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = self.preprocess(X, y, return_feature_columns=False)
        user_ctr = df.groupby(self.group_on_column)[self.target_column].mean().fillna(0).reset_index()
        user_ctr.columns = [self.group_on_column, 'ctr_per_user']
        df = df.merge(user_ctr, on=self.group_on_column, how='left')
        return df
