from sklearn.linear_model import LogisticRegression

from model_pipeline import ModelPipeline
from data_loader.base_data_loader import DataLoader
from preprocessor import Preprocessor


def train():
    data_loader = DataLoader(train_file='data/train_dataset_full.csv', target_column='is_click')
    data_loader.load_data()
    X_train, X_val, y_train, y_val = data_loader.split_data()
    preprocessor = Preprocessor()
    model_pipeline = ModelPipeline(preprocessor=preprocessor, model=LogisticRegression(), params={
        'model__C': [0.001, 0.01, 0.1, 1, 10],
        'model__penalty': ['l2'],
        'model__class_weight': ['balanced'],
        'model__solver': ['lbfgs'],
        'model__tol': [1e-4, 1e-3]
    })
    model_pipeline.grid_search(X_train, y_train)
    print(f"Best model {model_pipeline.best_model}")

if __name__ == '__main__':
    train()