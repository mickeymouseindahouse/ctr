import json
from typing import List
import wandb
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import os
import sys
from pickle_object import PickleObject
from constants import PR0JECT_NAME



class BaseModelPipeline(PickleObject):
    def __init__(self, steps: List[tuple], grid_search_params: dict[str, dict]=None, scoring=log_loss, random_state=42, result_path: str = ''):
        """
        Initialize the pipeline.

        Args:
            steps: list of PicketObjects like Preprocessors or Models
            grid_search_params: dict of parameters to pass to the pipeline. key is class name, value is param dict for class
        """
        super().__init__(result_path)
        self.pipeline = Pipeline(steps)
        self.grid_search_params = {
            f"{step_name}__{param_name}": values
            for step_name, params in (grid_search_params or {}).items()
            for param_name, values in params.items()
        }
        self.scoring = scoring  # Now using Logloss
        self.best_model = None
        self.best_score = None
        self.best_params = None

    def fit(self, X_train, y_train):
        """Train the model."""
        self.pipeline.fit(X_train, y_train)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X, y_train=None):
        self.fit(X, y_train)
        return self.transform(X)

    def predict(self, X):
        """Make predictions."""
        return self.pipeline.predict(X)

    def fit_predict(self, X, y):
        """Make predictions."""
        self.fit(X, y)
        return self.predict(X)

    def score(self, X, y, **kwargs):
        """Custom scoring method."""
        y_pred = self.predict(X)
        return self.scoring(y, y_pred, **kwargs)

    def evaluate(self, X, y):
        """Evaluate the model on test data."""
        return self.score(X, y)

    @property
    def _estimator_type(self):
        # Delegate to the wrapped model's estimator type (e.g., 'classifier' or 'regressor')
        return self.model._estimator_type


    def grid_search(self, X_train, y_train, cv=5, verbose=5, n_jobs=10):
        """Perform grid search to optimize hyperparameters."""
        print("Starting GridSearchCV")
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid=self.grid_search_params,
            cv=cv,
            scoring=self.scoring,  # Use the stored scoring function (e.g., f1_score)
            verbose=verbose,
            n_jobs=n_jobs
        )
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        self.best_score = grid_search.best_score_
        print(f"GridSearchCV complete with best score: {self.best_score}")
        return self.best_score

    def dump_results(self, class_name: str = None, results: str = ''):
        results = f"best_params = {self.best_params}\n"
        results += f"best_model = {self.best_model}\n"
        results += f"best_score = {self.best_score}\n"
        super().dump_results(results=results)

        """
        import joblib
        joblib.dump(self, "base_model_pipeline.joblib")
        artifact = wandb.Artifact("model", type="pipeline")
        artifact.add_file("base_model_pipeline.joblib")
        wandb.log_artifact(artifact)
        """


