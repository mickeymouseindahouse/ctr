import os.path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from constants import getroot
from data_loader.train_loader_session_splitter import TrainLoaderSessionSplitter
from model.base_model import BaseModel
from pipeline.base_model_pipeline import BaseModelPipeline
from preprocessor.fill_na_preprocessor import FillNaPreprocessor
from preprocessor.scaler_encoder_preprocessor import ScalerEncoderPreprocessor

if __name__ == '__main__':
    data_loader = TrainLoaderSessionSplitter(result_path="Micha001", train_file=os.path.join(getroot(), "data/train_dataset_full.csv"),
                                             preprocessing=BaseModelPipeline(steps=[FillNaPreprocessor()]))
    data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.split_data()
    data_loader.dump_to_pickle()

    pipeline = BaseModelPipeline(result_path="Micha001", steps=[
        ScalerEncoderPreprocessor(
            one_hot_features=['gender', 'product'],
            label_features=['user_depth'],
        ),
        BaseModel(model=DecisionTreeClassifier()),],
        grid_search_params={"BaseModel": {"model__min_samples_leaf": [1, 2, 3],}},)
    # pipeline.fit(X_train, y_train)
    pipeline.grid_search(X_train, y_train)
    print(pipeline.best_params)
    pipeline.dump_to_pickle()
