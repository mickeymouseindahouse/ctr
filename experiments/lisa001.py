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
    data_loader = TrainLoaderSessionSplitter(result_path="lisa001", train_file=os.path.join(getroot(), "data/train_dataset_full.csv"),
                                             preprocessing=BaseModelPipeline(steps=[FillNaPreprocessor()]))
    data_loader.load_data()