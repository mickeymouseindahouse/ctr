import os
import dill as pickle
from datetime import datetime


class PickleObject:
    def __init__(self, result_path: str = ''):
        self.result_path = result_path

    def dump_to_pickle(self, class_name: str = None) -> None:
        """
        Save the fitted preprocessor to a pickle file.
        """
        file_path = os.path.join(os.getenv('PROJECT_ROOT'), 'results', self.result_path,
                                 f'{class_name or self.__class__.__name__}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def dump_results(self, class_name: str = None, results: str = ''):
        file_path = os.path.join(os.getenv('PROJECT_ROOT'), 'results', self.result_path,
                                 f'CV-{class_name or self.__class__.__name__}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(results)

    @staticmethod
    def load_pickle(self, file_name) -> 'PickleObject':
        return pickle.load(open(file_name, "rb"))
