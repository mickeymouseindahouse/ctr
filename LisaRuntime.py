from sklearn.linear_model import LogisticRegression

from data_loader import DataLoader
from model_pipeline import ModelPipeline
from preprocessor import Preprocessor


def main():
    data_loader = DataLoader(train_file='data/train_dataset_full.csv')
    data_loader.load_data()
    # maybe some of the preprocessing should be done before splitting?
    X_train, X_test, y_train, y_test = data_loader.split_data(test_size=0.2)
    model_pipeline = ModelPipeline(preprocessor=Preprocessor(), model=LogisticRegression(), params={
        'model__C': [0.001, 0.01, 0.1, 1, 10],
        'model__penalty': ['l2'],
        'model__class_weight': ['balanced'],
        'model__solver': ['lbfgs'],
        'model__tol': [1e-4, 1e-3]
    })
    model_pipeline.grid_search(X_train, y_train)
    print(f"best estimator {model_pipeline.best_model}")


if __name__ == '__main__':
    main()

