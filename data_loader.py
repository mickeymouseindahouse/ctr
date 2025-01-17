import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, train_file: str, test_file: str = None, target_column: str = None):
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
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Load training and test data from CSV files."""
        self.train_data = pd.read_csv(self.train_file)
        if self.test_file:
            self.test_data = pd.read_csv(self.test_file)
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

        session_timestamps = self.train_data.groupby('session_id')["DateTime"].min().reset_index()

        session_timestamps = session_timestamps.sort_values(by='DateTime')

        train_size = int(len(session_timestamps) * (1.0 - test_size))

        train_sessions = session_timestamps['session_id'].iloc[:train_size]
        test_sessions = session_timestamps['session_id'].iloc[train_size:]

        train_df = self.train_data[self.train_data['session_id'].isin(train_sessions)]
        test_df = self.train_data[self.train_data['session_id'].isin(test_sessions)]

        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_df.drop(columns=['is_click']), test_df.drop(columns=['is_click']), train_df[['is_click']], test_df[
            ['is_click']]
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_train_data(self):
        """Retrieve the training data."""
        return self.X_train, self.y_train

    def get_validation_data(self):
        """Retrieve the validation data."""
        return self.X_test, self.y_test

    def get_test_data(self):
        """Retrieve the test data (if provided)."""
        if self.test_data is None:
            raise ValueError("Test data not loaded. Provide a test file path during initialization.")
        return self.test_data

# Example usage:
if __name__ == "__main__":
    # Initialize DataLoader
    data_loader = DataLoader("train.csv", "test.csv", target_column="target")

    # Load data
    data_loader.load_data()

    # Split data into training and validation sets
    data_loader.split_data(test_size=0.2, random_state=42)

    # Retrieve data
    X_train, y_train = data_loader.get_train_data()
    X_val, y_val = data_loader.get_validation_data()
    test_data = data_loader.get_test_data()

    print("Training data:", X_train.shape)
    print("Validation data:", X_val.shape)
    print("Test data:", test_data.shape if test_data is not None else "No test data")
