import os.path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from constants import getroot
from data_loader.train_loader_session_splitter import TrainLoaderSessionSplitter
from model.base_model import BaseModel
from pipeline.base_model_pipeline import BaseModelPipeline
from preprocessor.fill_na_preprocessor import FillNaPreprocessor
from preprocessor.scaler_encoder_preprocessor import ScalerEncoderPreprocessor

if __name__ == '__main__':
    data_loader = TrainLoaderSessionSplitter(results_path="Micha001", train_file=os.path.join(getroot(), "data/train_dataset_full.csv"),
                                             preprocessing=BaseModelPipeline(steps=[FillNaPreprocessor()]))
    data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.split_data()
    data_loader.dump_to_pickle()

    pipeline = BaseModelPipeline(results_path="Micha001", steps=[
        ScalerEncoderPreprocessor(
            one_hot_features=['gender', 'product'],
            label_features=['user_depth']
        ),
        BaseModel(model=DecisionTreeClassifier()),],
        grid_search_params={"LogRegModel": {"C": [0.1, 1, 10, 1000, 1000], "max_iter": [1000000, 10000000]},})
    preds = pipeline.fit_predict(X_train, y_train)
    pipeline.dump_to_pickle()
    print(np.unique(preds))
