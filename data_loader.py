import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, train_file: str, test_file: str = None, target_column: str = 'is_click', date_column: str = 'DateTime'):
        """
        Initialize the DataLoader.

        Args:
            train_file (str): Path to the training CSV file.
            test_file (str): Path to the test CSV file (optional).
            target_column (str): Name of the target column (optional).
        """
        self.train_file = train_file
        self.test_file = test_file
        self.target_column = target_column
        self.train_data = None
        self.test_data = None
        self.date_column = date_column

    def load_data(self):
        """Load training and test data from CSV files."""
        self.train_data = pd.read_csv(self.train_file, parse_dates=[self.date_column])
        if self.test_file:
            self.test_data = pd.read_csv(self.test_file, parse_dates=[self.date_column])
        else:
            self.test_data = None
        self._fill_na()


    def _fill_na(self):
        self.train_data.dropna(inplace=True)

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

        session_timestamps = self.train_data.groupby('session_id')["DateTime"].min().reset_index()

        session_timestamps = session_timestamps.sort_values(by='DateTime')

        train_size = int(len(session_timestamps) * (1.0 - test_size))

        train_sessions = session_timestamps['session_id'].iloc[:train_size]
        test_sessions = session_timestamps['session_id'].iloc[train_size:]

        train_df = self.train_data[self.train_data['session_id'].isin(train_sessions)]
        test_df = self.train_data[self.train_data['session_id'].isin(test_sessions)]

        train_df["day"] = train_df.DateTime.dt.day
        test_df["day"] = test_df.DateTime.dt.day
        train_df.drop(["DateTime"], axis=1, inplace=True)
        test_df.drop(["DateTime"], axis=1, inplace=True)

        X_train, X_test, y_train, y_test =\
            train_df.drop(columns=['is_click']), test_df.drop(columns=['is_click']), train_df[['is_click']].values, test_df[
            ['is_click']].values
        return X_train, X_test, y_train, y_test