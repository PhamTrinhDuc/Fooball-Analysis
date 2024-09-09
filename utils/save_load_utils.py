import pickle
from configs.config import APP_CFG

def save_data(path: str, data) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_data(path: str) :
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data