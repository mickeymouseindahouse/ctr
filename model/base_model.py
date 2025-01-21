from typing import Optional, Type

from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from pickle_object import PickleObject


class BaseModel(PickleObject, BaseEstimator):
    def __init__(self, model: BaseEstimator,
                 grid_search_params:Optional[dict] = None,
                 score=f1_score, random_state=42, results_path: str = ''):
        """
        Base model class.

        Args:
            model: The BaseEstimator implementer model instance.
            grid_search_params:Optional[dict]: params for grid search
            score: Callable score function.
            random_state: Random state for model fitting.
        """
        super().__init__(results_path)
        self.model = model
        self.grid_search_params = grid_search_params or {}
        self.score = score
        self.random_state = random_state
        self.best_model = None
        self.best_score = None
        self.best_params = None

    def fit(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

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
        grid_search = GridSearchCV(self.model, self.grid_search_params, cv=cv, scoring=self.score)
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        self.best_score, self.best_params = grid_search.best_score_, grid_search.best_params_
        return self.best_score, self.best_params

    def dump_to_pickle(self, class_name: str = None) -> None:
        super().dump_to_pickle(self.model.__class__.__name__)
