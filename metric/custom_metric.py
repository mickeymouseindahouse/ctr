from catboost import CatBoostClassifier, Pool
import numpy as np

class CustomCostMetric(object):
    def is_max_optimal(self):
        return False  # smaller is better

    def __init__(self, fp_cost=5, fn_cost=10):
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost

    def evaluate(self, approxes, target, weight):
        preds = (approxes[0] > 0.5).astype(int)
        fp = np.sum((preds == 1) & (target == 0))
        fn = np.sum((preds == 0) & (target == 1))

        total_cost = (fp * self.fp_cost) + (fn * self.fn_cost)
        return total_cost, 1