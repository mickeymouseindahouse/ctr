import sys
sys.path.append("/content/ctr")
import os.path
import pandas as pd
import numpy as np
import os


from constants import getroot
from catboost import CatBoostClassifier
from data_loader.train_loader_session_splitter import TrainLoaderSessionSplitter
from model.base_model import BaseModel
from pipeline.base_model_pipeline import BaseModelPipeline
from preprocessor.ctr_ratio_label_preprocessor import RatioBasedLabelPreprocessor
from preprocessor.fill_na_preprocessor import FillNaPreprocessor, FillAlgo
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

RESULT_PATH = 'first_submission'

if __name__ == '__main__':
    # Define numerical and categorical columns
    numerical_cols = ['city_development_index', 'age_level', 'user_depth']
    categorical_cols = ['gender', 'product', 'campaign_id']

    # Custom median function
    def custom_median(series: pd.Series) -> float:
        return series.median()

    # Create preprocessors
    num_preprocessor = FillNaPreprocessor(
        columns=numerical_cols,
        fill_algo=FillAlgo.CUSTOM,
        custom_function=custom_median
    )
    cat_preprocessor = FillNaPreprocessor(
        columns=categorical_cols,
        fill_algo=FillAlgo.MODE
    )

    data_loader = TrainLoaderSessionSplitter(
        result_path=RESULT_PATH,
        train_file=os.path.join(getroot(), "data/train_dataset_full.csv"),
        preprocessing=BaseModelPipeline(
            steps=[
                ("numerical_imputer", num_preprocessor),
                ("categorical_imputer", cat_preprocessor)
            ]
        )
    )

    # Load the raw data
    raw_data = pd.read_csv(os.path.join(getroot(), "data/train_dataset_full.csv"))
    print("Raw data columns:", raw_data.columns)

    data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.split_data()
    data_loader.dump_to_pickle()

    # Convert y_train and y_test to 1 and -1 for CatBoost
    y_train = np.where(y_train == 0, -1, 1)  # Convert 0 to -1, 1 remains 1
    y_test = np.where(y_test == 0, -1, 1)    # Convert 0 to -1, 1 remains 1

    # Calculate class weights without using np.bincount
    class_counts = {
        -1: np.sum(y_train == -1),  # Count of class -1
        1: np.sum(y_train == 1)     # Count of class 1
    }
    total_samples = len(y_train)
    class_weights = {
        -1: total_samples / (2 * class_counts[-1]),  # Weight for class -1
        1: total_samples / (2 * class_counts[1])    # Weight for class 1
    }

    # Ensure categorical features are strings
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    print(X_train[['gender', 'product', 'campaign_id', 'webpage_id']].dtypes)

    # Create a valid scorer for F1 score
    f1_scorer = make_scorer(f1_score)

    pipeline = BaseModelPipeline(
        result_path=RESULT_PATH,
        steps=[
            ("encoder", RatioBasedLabelPreprocessor(
                label_features=['gender', 'product', 'campaign_id']
            )),
            ("classifier", BaseModel(model=CatBoostClassifier(
                verbose=0,
                cat_features=['gender', 'product', 'campaign_id'],
                class_weights=class_weights,  # Use class_weights instead of scale_pos_weight
                loss_function='Logloss'  # Explicitly set loss function to Logloss
            )))
        ],
        grid_search_params={
            "classifier": {
                "model__depth": [4, 6, 8],
                "model__iterations": [300, 500],
                "model__learning_rate": [0.03, 0.1],
                "model__l2_leaf_reg": [1, 5, 10]
            }
        },
        scoring=f1_scorer  # Use the valid F1 scorer
    )

    # Train with F1 scoring
    pipeline.grid_search(X_train, y_train, cv=3)
    pipeline.dump_to_pickle()
    pipeline.dump_results()
