from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd

class ModelPipeline:
    def __init__(self, preprocessor, model, params):
        """
        Initialize the pipeline.

        Args:
            preprocessor: Preprocessing transformer (e.g., ColumnTransformer).
            model: The machine learning model (e.g., LogisticRegression, BernoulliNB).
            params (dict): Hyperparameters for grid search.
        """
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        self.params = params
        self.best_model = None

    def grid_search(self, X, y, cv=5):
        """
        Perform grid search with cross-validation.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target values.
            cv (int): Number of cross-validation folds.

        Returns:
            dict: Best parameters found.
        """
        grid_search = GridSearchCV(self.pipeline, self.params, cv=cv, scoring="accuracy")
        grid_search.fit(X, y)
        self.best_model = grid_search.best_estimator_
        return grid_search.best_params_

    def predict(self, X):
        """
        Make predictions using the best model.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted values.
        """
        if not self.best_model:
            raise ValueError("Model has not been fitted. Perform grid search first.")
        return self.best_model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluate the model.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): True labels.

        Returns:
            float: Accuracy score.
        """
        y_pred = self.predict(X)
        return f1_score(y, y_pred)
