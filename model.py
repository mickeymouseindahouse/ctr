from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

class Model:
    def __init__(self, model, params=None):
        """
        Base model class.

        Args:
            model: The sklearn model instance.
            params (dict): Hyperparameters for grid search.
        """
        self.model = model
        self.params = params
        self.best_model = None

    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Evaluate the model on test data."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def grid_search(self, X_train, y_train, cv=5, scoring='f1'):
        """Perform grid search to optimize hyperparameters."""
        if not self.params:
            raise ValueError("No parameters specified for grid search.")
        grid_search = GridSearchCV(self.model, self.params, cv=cv, scoring=scoring)
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        return grid_search.best_params_
