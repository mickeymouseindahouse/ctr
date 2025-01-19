import pickle

import pandas as pd


class BasePreprocessor:
    # def __init__(self, numerical_features=None, cat_features=None):
    #     """
    #     Initialize the Preprocessor.
    #
    #     Args:
    #        numerical_features (list): List of numerical column names to be scaled.
    #        cat_features (list): List of categorical column names to be encoded.
    #     """
    #     if cat_features is None:
    #         cat_features = ['product', 'gender']
    #     self.numerical_features = numerical_features if numerical_features else []
    #     self.cat_features = cat_features if cat_features else []
    #     self.preprocessor = None

    def __init__(self):
        self.preprocessor = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the preprocessor to the data.

        Args:
            X (pd.DataFrame): The input data to fit the preprocessor on.
            y (pd.Series): ignored
        """
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """
                Transform the input data using the fitted preprocessor.

                Args:
                    X (pd.DataFrame): The input data to transform.

                Returns:
                    pd.DataFrame: The transformed data.
                """
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the input data.

        Args:
            X (pd.DataFrame): The input data to fit and transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def save_to_pickle(self, file_path: str):
        """
        Save the fitted preprocessor to a pickle file.

        Args:
            file_path (str): Path to the pickle file to save the preprocessor.
        """
        if not self.preprocessor:
            raise ValueError("Preprocessor has not been fitted. Call fit() before saving.")

        with open(file_path, "wb") as f:
            pickle.dump(self.preprocessor, f)

    @staticmethod
    def load_from_pickle(file_path: str):
        """
        Load a preprocessor from a pickle file.

        Args:
            file_path (str): Path to the pickle file.

        Returns:
            Preprocessor: The loaded preprocessor.
        """
        with open(file_path, "rb") as f:
            loaded_preprocessor = pickle.load(f)

        return loaded_preprocessor