import json
from typing import List

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from pickle_object import PickleObject


class BaseModelPipeline(PickleObject):
    def __init__(self, steps: List[PickleObject], grid_search_params: dict[str, dict]=None, score=f1_score, random_state=42, results_path: str = ''):
        """
        Initialize the pipeline.

        Args:
            steps: list of PicketObjects like Preprocessors or Models
            grid_search_params: dict of parameters to pass to the pipeline. key is class name, value is param dict for class
        """
        super().__init__(results_path)
        self.pipeline = Pipeline([
            (step.__class__.__name__, step) for step in steps])
        self.grid_search_params = {f"{class_name}__{k}": v for class_name, item_dict in grid_search_params.items()
                       for k, v in item_dict.items()} if grid_search_params else None
        self.score = score
        self.best_model = None
        self.best_score = None
        self.best_params = None

    def fit(self, X_train, y_train):
        """Train the model."""
        self.pipeline.fit(X_train, y_train)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X):
        self.fit(X, y_train=None)
        return self.transform(X)

    def predict(self, X):
        """Make predictions."""
        return self.pipeline.predict(X)

    def fit_predict(self, X, y):
        """Make predictions."""
        self.fit(X, y)
        return self.predict(X)

    def evaluate(self, X, y):
        """Evaluate the model on test data."""
        y_pred = self.predict(X)
        return self.score(y, y_pred)

    def grid_search(self, X_train, y_train, cv=5):
        """Perform grid search to optimize hyperparameters."""
        grid_search = GridSearchCV(self.pipeline, self.grid_search_params, cv=cv, scoring=self.score)
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        self.best_score, self.best_params = grid_search.best_score_, grid_search.best_params_
        return self.best_score, self.best_params

