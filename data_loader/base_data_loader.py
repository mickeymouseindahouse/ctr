from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from pickle_object import PickleObject
from pipeline.base_model_pipeline import BaseModelPipeline


class BaseDataLoader(PickleObject):
    def __init__(self, train_file: str, test_file: str = None, target_column: str = 'is_click',
                 date_columns: Tuple[str, ...] = ('DateTime',), preprocessing: BaseModelPipeline = None, result_path: str = ''):
        """
        Initialize the DataLoader.

        Args:
            train_file (str): Path to the training CSV file.
            test_file (str): Path to the test CSV file (optional).
            target_column (str): Name of the target column (optional).
            date_columns (Tuple[str, ...]): Tuple of column names to parse as date columns.
            preprocessing: BaseModelPipeline Preprocessing pipeline you wanna carry out before splitting
        """
        super().__init__(result_path)
        self.train_file = train_file
        self.test_file = test_file
        self.target_column = target_column
        self.train_data = None
        self.test_data = None
        self.date_columns = date_columns
        self.preprocessing = preprocessing

    def load_data(self):
        """Load training and test data from CSV files."""
        self.train_data = pd.read_csv(self.train_file, parse_dates=list(self.date_columns))
        self.train_data.columns = self.train_data.columns.str.lower()

        if self.test_file:
            self.test_data = pd.read_csv(self.test_file, parse_dates=list(self.date_columns))
            self.test_data.columns = self.test_data.columns.str.lower()
        else:
            self.test_data = None
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess train and test data if available."""
        if self.preprocessing is not None:
            self.train_data = self.preprocessing.fit_transform(X=self.train_data)
        if self.test_data is not None:
            self.test_data = self.preprocessing.transform(X=self.test_data)


    def split_data(self, test_size=0.2, random_state=42) -> tuple:
        """
        Split the training data into training and validation sets.

        Args:
            test_size (float): Proportion of the data to be used as validation data.
            random_state (int): Random state for reproducibility.
        """
        if self.train_data is None:
            raise ValueError("Train data not loaded. Call load_data() first.")

        if self.target_column is None:
            raise ValueError("Target column is not specified.")

        return train_test_split(self.train_data, self.target_column, test_size=test_size, random_state=random_state)