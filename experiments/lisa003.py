import os.path

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from constants import getroot
from data_loader.train_loader_session_splitter import TrainLoaderSessionSplitter
from model.base_model import BaseModel
from pipeline.base_model_pipeline import BaseModelPipeline
from preprocessor.ctr_ratio_label_preprocessor import RatioBasedLabelPreprocessor
from preprocessor.fill_na_preprocessor import FillNaPreprocessor
from preprocessor.sampler import Sampler
from sklearn.ensemble import RandomForestClassifier

RESULT_PATH = 'Lisa003'

if __name__ == '__main__':
    data_loader = TrainLoaderSessionSplitter(result_path=RESULT_PATH,
                                             train_file=os.path.join(getroot(), "data/train_dataset_full.csv"),
                                             preprocessing=BaseModelPipeline(steps=[Sampler(), FillNaPreprocessor()]))
    data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.split_data()
    data_loader.dump_to_pickle()

    pipeline = BaseModelPipeline(result_path=RESULT_PATH, steps=[
        RatioBasedLabelPreprocessor(
            one_hot_features=['gender'],
            label_features=['product', 'campaign_id','webpage_id','product_category_1', 'product_category_2']
        ),
        BaseModel(model=RandomForestClassifier(n_estimators=100)), ],
                                 grid_search_params={"BaseModel": {"model__min_samples_leaf": [ 1, 5, 10],
                                                                   "model__n_estimators": np.logspace(1, 2.5, 10, dtype=int),  }}, )
    pipeline.grid_search(X_train, y_train, cv=3, )
    pipeline.dump_to_pickle()
    pipeline.dump_results()
