import os.path
import numpy as np
from constants import getroot
from data_loader.train_loader_session_splitter import TrainLoaderSessionSplitter
from model.base_model import BaseModel
from pipeline.base_model_pipeline import BaseModelPipeline
from preprocessor.ctr_ratio_label_preprocessor import RatioBasedLabelPreprocessor
from preprocessor.fill_na_preprocessor import FillNaPreprocessor
from preprocessor.sampler import Sampler
from sklearn.ensemble import RandomForestClassifier

RESULT_PATH = 'best_cv'

if __name__ == '__main__':
    data_loader = TrainLoaderSessionSplitter(result_path=RESULT_PATH,
                                             train_file=os.path.join(getroot(), "data/train_dataset_full.csv"),
                                             preprocessing=BaseModelPipeline([FillNaPreprocessor()]))
    data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.split_data()
    rblp = RatioBasedLabelPreprocessor(
        one_hot_features=['gender'],
        label_features=['product', 'campaign_id', 'webpage_id', 'product_category_1', 'product_category_2']
    )
    X_train = rblp.fit_transform(X_train)

    data_loader.dump_to_pickle()

    pipeline = BaseModelPipeline.load_pickle(os.path.join(getroot(), "results/Michaa006/BaseModelPipeline-20250129_001725.pkl"))
    pipeline.grid_search(X_train, y_train, cv=3, )
    pipeline.dump_to_pickle()
    pipeline.dump_results()