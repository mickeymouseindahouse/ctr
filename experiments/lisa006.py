import os.path
import numpy as np
from constants import getroot
from data_loader.train_loader_session_splitter import TrainLoaderSessionSplitter
from pipeline.base_model_pipeline import BaseModelPipeline
from preprocessor.fill_na_preprocessor import FillNaPreprocessor
import pandas as pd

RESULT_PATH = 'Lisa006'

if __name__ == '__main__':
    data_loader = TrainLoaderSessionSplitter(result_path=RESULT_PATH,
                                             train_file=os.path.join(getroot(), "data/train_dataset_full.csv"),
                                             test_file=os.path.join(getroot(), "data/X_test_1st.csv"),
                                             preprocessing=BaseModelPipeline(steps=[FillNaPreprocessor()]))
    data_loader.load_data()
    X_train, X_val, y_train, y_val = data_loader.split_data()
    data_loader.dump_to_pickle()
    pipeline = BaseModelPipeline.load_pickle(os.path.join(getroot(), 'results/first_submission/BaseModelPipeline-20250201_225341.pkl'))
    print(pipeline.best_model.score(X_val, y_val))
    X_test = data_loader.test_data
    preds = pipeline.best_model.predict(X_test)
    print(np.bincount(np.array(preds, dtype=np.int64)))
    pd.Series(preds).to_csv(getroot() + '/results/first_submission/preds.csv', index=False, header=False)
