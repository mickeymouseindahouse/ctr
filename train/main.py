import os

from data_loader.base_data_loader import BaseDataLoader
from constants import setenv, getroot

if __name__ == '__main__':
    setenv()
    data_loader = BaseDataLoader(train_file=os.path.join(getroot(), "data/train_dataset_full.csv"))
    data_loader.load_data()
    data_loader.dump_to_pickle()