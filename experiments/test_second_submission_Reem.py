import os.path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from constants import getroot
from data_loader.train_loader_session_splitter import TrainLoaderSessionSplitter
from model.base_model import BaseModel
from pipeline.base_model_pipeline import BaseModelPipeline
from preprocessor.ctr_ratio_label_preprocessor import RatioBasedLabelPreprocessor
from preprocessor.fill_na_preprocessor import FillNaPreprocessor, FillAlgo
from sklearn.ensemble import RandomForestClassifier

from preprocessor.sampler import OverSampler, UnderSampler

RESULT_PATH = 'first_submission'

numerical_cols = ['city_development_index', 'age_level', 'user_depth']
categorical_cols = ['gender', 'product', 'campaign_id']


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
test_file = os.path.join(getroot(), "data/X_test_1st.csv"),
preprocessing=BaseModelPipeline(
    steps=[
        ("numerical_imputer", num_preprocessor),
        ("categorical_imputer", cat_preprocessor)
    ]
)
)

data_loader.load_data()
X_train, X_val, y_train, y_val = data_loader.split_data()
data_loader.dump_to_pickle()

# Convert y_train and y_test to 1 and -1 for CatBoost
y_train = np.where(y_train == 0, -1, 1)  # Convert 0 to -1, 1 remains 1
y_val = np.where(y_val == 0, -1, 1)    # Convert 0 to -1, 1 remains 1

# Ensure categorical features are strings
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str)
    X_val[col] = X_val[col].astype(str)

pipeline = BaseModelPipeline.load_pickle(os.path.join(getroot(), 'results/second_submission/f1.pkl'))
print(pipeline.best_params)
print(pipeline.best_model.score(X_val, y_val))

X_test = data_loader.test_data
X_test.drop('datetime', axis=1, inplace=True)

# Ensure categorical features are strings
for col in categorical_cols:
    X_test[col] = X_test[col].astype(str)

preds = pipeline.best_model.predict(X_test)
preds = np.where(preds == -1, 0, 1)    # Convert 0 to -1, 1 remains 1
print(np.bincount(np.array(preds, dtype=np.int64)))

pd.Series(preds).to_csv(getroot() + '/results/second_submission/y_preds.csv', index=False, header=False)

from sklearn.metrics import confusion_matrix, f1_score
y_test = pd.read_csv(getroot() + '/data/y_test_1st.csv', header=None)
f1 = f1_score(y_test, preds)
print(f'F1 Score on Test Set: {f1:.3f}')

print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))

# Convert predictions to 0 and 1
y_pred = np.where(preds == -1, 0, 1)

# Recalculate CTR rate
ctr_rate = preds.mean()
print("Predicted CTR Rate:", ctr_rate)

# Evaluate additional metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score

precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
roc_auc = roc_auc_score(y_test, preds)

print("Precision:", precision)
print("Recall:", recall)
print("ROC-AUC:", roc_auc)
