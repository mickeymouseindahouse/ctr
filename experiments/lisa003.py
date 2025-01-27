import os.path
from sklearn.tree import DecisionTreeClassifier
from constants import getroot
from data_loader.train_loader_session_splitter import TrainLoaderSessionSplitter
from model.base_model import BaseModel
from pipeline.base_model_pipeline import BaseModelPipeline
from preprocessor.ctr_ratio_label_preprocessor import RatioBasedLabelPreprocessor
from preprocessor.fill_na_preprocessor import FillNaPreprocessor
from preprocessor.undersampler import Undersampler

RESULT_PATH = 'Lisa003'

if __name__ == '__main__':
    data_loader = TrainLoaderSessionSplitter(result_path=RESULT_PATH,
                                             train_file=os.path.join(getroot(), "data/train_dataset_full.csv"),
                                             preprocessing=BaseModelPipeline(steps=[Undersampler()]))
    data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.split_data()
    data_loader.dump_to_pickle()

    pipeline = BaseModelPipeline(result_path=RESULT_PATH, steps=[
        RatioBasedLabelPreprocessor(
            one_hot_features=['gender'],
            label_features=['product', 'campaign_id','webpage_id','product_category_1', 'product_category_2']
        ),
        BaseModel(model=DecisionTreeClassifier(class_weight='balanced')), ],
                                 grid_search_params={"BaseModel": {"model__min_samples_leaf": [2, 1, 3], }}, )
    pipeline.grid_search(X_train, y_train)
    pipeline.dump_to_pickle()
    pipeline.dump_results()
