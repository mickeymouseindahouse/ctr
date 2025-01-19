import os
import pickle
from datetime import datetime


class PickleObject:
    def dump_to_pickle(self, ) -> None:
        """
        Save the fitted preprocessor to a pickle file.
        """
        file_path = f'{os.getenv('PROJECT_ROOT')}/results/{self.__class__.__name__}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl'
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickle(self, file_name) -> 'PickleObject':
        return pickle.load(open(file_name, "rb"))
