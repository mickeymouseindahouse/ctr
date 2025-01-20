import os.path

from constants import getroot
from data_loader.train_loader_session_splitter import TrainLoaderSessionSplitter
from model.LogRegModel import LogRegModel
from pipeline.base_model_pipeline import BaseModelPipeline
from preprocessor.fill_na_preprocessor import FillNaPreprocessor
from preprocessor.scaler_encoder_preprocessor import ScalerEncoderPreprocessor

if __name__ == '__main__':
    data_loader = TrainLoaderSessionSplitter(train_file=os.path.join(getroot(), "data/train_dataset_full.csv"),
                                             preprocessing=BaseModelPipeline(steps=[FillNaPreprocessor()]))
    data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.split_data()

    pipeline = BaseModelPipeline(steps=[
        FillNaPreprocessor(),
        ScalerEncoderPreprocessor(
            one_hot_features=['gender', 'product'],
            label_features=['user_depth']
        ),
        LogRegModel()],
        params={"LogRegModel": {"C": [0.1, 1, 10]}})
    pipeline.fit(X_train, y_train)