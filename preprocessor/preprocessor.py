import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Preprocessor:
    def __init__(self, numerical_features=None, cat_features=None):
        """
        Initialize the Preprocessor.

        Args:
           numerical_features (list): List of numerical column names to be scaled.
           cat_features (list): List of categorical column names to be encoded.
        """
        if cat_features is None:
            cat_features = ['product', 'gender']
        self.numerical_features = numerical_features if numerical_features else []
        self.cat_features = cat_features if cat_features else []
        self.preprocessor = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the preprocessor to the data.

        Args:
            X (pd.DataFrame): The input data to fit the preprocessor on.
        """
        self.numerical_features = [
            col for col in X.columns if col not in self.cat_features
        ]

        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.numerical_features),
                ("cat", categorical_transformer, self.cat_features),
            ]
        )

        # Fit the preprocessor on the data
        self.preprocessor.fit(X)
        return self


    def transform(self, X: pd.DataFrame, y=None):
        """
                Transform the input data using the fitted preprocessor.

                Args:
                    X (pd.DataFrame): The input data to transform.

                Returns:
                    pd.DataFrame: The transformed data.
                """
        if not self.preprocessor:
            raise ValueError("Preprocessor has not been fitted. Call fit() before transform().")

        transformed_data = self.preprocessor.transform(X)
        return pd.DataFrame(transformed_data)

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