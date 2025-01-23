import os

SCALER = "scaler"
ONE_HOT_ENCODER = "one_hot_encoder"
LABEL_ENCODER = "label_encoder"
CTR_LABEL_ENCODER = "ctr_label_encoder"

def setenv():
    os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.abspath(__file__))

def getroot() -> str:
    if os.getenv("PROJECT_ROOT") is None:
        setenv()
    return os.getenv("PROJECT_ROOT")