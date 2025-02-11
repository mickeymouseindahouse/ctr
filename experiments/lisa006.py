import os.path
import numpy as np
from catboost import CatBoostClassifier

from constants import getroot
from data_loader.train_loader_session_splitter import TrainLoaderSessionSplitter
from model.base_model import BaseModel
from pipeline.base_model_pipeline import BaseModelPipeline
from preprocessor.ctr_per_user_encoder import CTRPerUserEncoder
from preprocessor.ctr_ratio_label_preprocessor import RatioBasedLabelPreprocessor
from preprocessor.fill_na_preprocessor import FillNaPreprocessor
from preprocessor.pca_preprocessor import PcaPreprocessor
from preprocessor.sampler import OverSampler, UnderSampler

RESULT_PATH = 'Lisa006'

if __name__ == '__main__':
    data_loader = TrainLoaderSessionSplitter(result_path=RESULT_PATH,
                                             train_file=os.path.join(getroot(), "data/train_dataset_full.csv"),
                                             preprocessing=BaseModelPipeline(steps=[UnderSampler(), FillNaPreprocessor(), CTRPerUserEncoder()]))
    data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.split_data()
    data_loader.dump_to_pickle()

    pipeline = BaseModelPipeline(result_path=RESULT_PATH, steps=[
        RatioBasedLabelPreprocessor(
            one_hot_features=['gender'],
        ),
        PcaPreprocessor(),
        BaseModel(model=CatBoostClassifier(verbose=0)), ],
                                 grid_search_params={ "BaseModel": {
                                            "model__depth": [6, 8, ],
                                            "model__iterations": [100, 300, ],
                                            "model__learning_rate": [0.01, 0.1, 0.2],
                                            "model__l2_leaf_reg": [1, 3, 5]
        }}, )
    pipeline.grid_search(X_train, y_train, cv=3, )
    pipeline.dump_to_pickle()
    pipeline.dump_results()