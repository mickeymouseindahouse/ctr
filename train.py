from ModelPipeline import ModelPipeline
from data_loader import DataLoader


def train():
    data_loader = DataLoader(train_file='train_dataset_full.csv', target_column='is_click')
    data_loader.load_data()
    X_train, X_val, y_train, y_val = data_loader.split_data()
    model_pipeline = ModelPipeline()

if __name__ == '__main__':
    pass