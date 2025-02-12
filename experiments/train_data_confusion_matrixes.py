import os.path
from constants import getroot
from data_loader.train_loader_session_splitter import TrainLoaderSessionSplitter
import matplotlib.pyplot as plt
from pipeline.base_model_pipeline import BaseModelPipeline
import numpy as np
import seaborn as sns
from preprocessor.fill_na_preprocessor import FillNaPreprocessor
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
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

        cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        cm_percent = np.nan_to_num(cm_percent, nan=0.0)
        accuracy = accuracy_score(df['y_true'], df['y_pred'])
        f1 = f1_score(df['y_true'], df['y_pred'], average='weighted')

        labels = np.array([[f"{count} ({percent:.1f}%)" for count, percent in zip(row_count, row_percent)]
                           for row_count, row_percent in zip(cm, cm_percent)])


        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", xticklabels=np.unique(y_test),
                    yticklabels=np.unique(y_test))

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix for Product: {product}\nAccuracy: {accuracy:.2f}, F1-score: {f1:.2f}')


        filename = os.path.join(save_dir, f'confusion_matrix_{product}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'Saved confusion matrix for {product} at {filename}')





