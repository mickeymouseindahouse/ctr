import pandas as pd
from sklearn.model_selection import train_test_split

class BaseDataLoader:
    def __init__(self, train_file: str, test_file: str = None, target_column: str = 'is_click',
                 date_columns: tuple[str] = 'DateTime'):
        """
        Initialize the DataLoader.

        Args:
            train_file (str): Path to the training CSV file.
            test_file (str): Path to the test CSV file (optional).
            target_column (str): Name of the target column (optional).
            date_columns (tuple[str]): Tuple of column names to parse as date columns.
        """
        self.train_file = train_file
        self.test_file = test_file
        self.target_column = target_column
        self.train_data = None
        self.test_data = None
        self.date_columns = date_columns

    def load_data(self):
        """Load training and test data from CSV files."""
        self.train_data = pd.read_csv(self.train_file, parse_dates=self.date_columns)
        if self.test_file:
            self.test_data = pd.read_csv(self.test_file, parse_dates=self.date_columns)
        else:
            self.test_data = None

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