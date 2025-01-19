from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from model.base_model import BaseModel


class LogRegModel(BaseModel):
    def __init__(self, model, params=None, score=f1_score, random_state=42):
        super().__init__(model=LogisticRegression(), params=params, score=score, random_state=random_state)
