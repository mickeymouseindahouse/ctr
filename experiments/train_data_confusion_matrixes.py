import os.path
from sklearn.metrics import confusion_matrix
from constants import getroot
from data_loader.train_loader_session_splitter import TrainLoaderSessionSplitter
import matplotlib.pyplot as plt
from pipeline.base_model_pipeline import BaseModelPipeline
import numpy as np
import seaborn as sns
from preprocessor.fill_na_preprocessor import FillNaPreprocessor

RESULT_PATH = 'train_data_confusion_matrixes.'

if __name__ == '__main__':
    data_loader = TrainLoaderSessionSplitter(result_path=RESULT_PATH,
                                             train_file=os.path.join(getroot(), "data/train_dataset_full.csv"),
                                             test_file=os.path.join(getroot(), "data/X_test_1st.csv"),
                                             preprocessing=BaseModelPipeline(steps=[FillNaPreprocessor()]))
    data_loader.load_data()
    pipeline = BaseModelPipeline.load_pickle(os.path.join(getroot(), 'results/first_submission/BaseModelPipeline-20250212_174202.pkl'))
    X_train, X_test, y_train, y_test = data_loader.split_data()
    data_loader.dump_to_pickle()
    preds = pipeline.best_model.predict(X_test)
    X_test = X_test.copy()
    X_test['y_pred'] = preds
    X_test['y_true'] = y_test
    save_dir = '/Users/lisapolotskaia/Downloads/cms/'
    os.makedirs(save_dir, exist_ok=True)

    for product in X_test['product'].unique():
        df = X_test[X_test['product'] == product]
        print(product)
        cm = confusion_matrix(df['y_true'], df['y_pred'])
        cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100  # Convert to percentage
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=np.unique(y_test),
                    yticklabels=np.unique(y_test))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix for Product: {product}')

        filename = os.path.join(save_dir, f'confusion_matrix_{product}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved confusion matrix for {product} at {filename}')







