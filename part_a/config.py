from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bank-additional-full.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
TEST_FRACTION = 0.20          # последние 20% по времени
N_EXPANDING_FOLDS = 3         # ≥3 проверки с расширяющимся окном

TARGET = "y"
# duration исключаем: неизвестна до завершения звонка
DROP_COLS = ["duration"]

# гиперпространства поиска (примеры; их можно расширять)
PARAM_GRIDS = {
    "logreg": {
        "C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3],
        "class_weight": [None, "balanced"],
    },
    "knn": {
        "n_neighbors": [3, 5, 11, 21],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "svm": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", 0.1],
    },
    "tree": {
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_leaf": [1, 5, 20, 50],
        "criterion": ["gini", "entropy"],
    },
    "rf": {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 5, 20],
        "max_features": ["sqrt", 0.5],
    },
    "gb": {
        "max_iter": [200, 400, 800],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, None],
        "min_samples_leaf": [1, 10, 30],
    },
    "nb": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    },
    "own_logreg": {
        "C": [1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6],   # сдвигаем вверх
        "lr": [0.05],                                   # меньше шаг
        "max_iter": [1000],
    },
}

NN_CONFIGS = [
    {"hidden": [128, 64],         "lr": 1e-3, "batch": 256, "dropout": 0.2, "wd": 1e-4},
    {"hidden": [256, 128, 64],    "lr": 1e-3, "batch": 256, "dropout": 0.3, "wd": 1e-4},
    {"hidden": [128, 64],         "lr": 5e-4, "batch": 128, "dropout": 0.1, "wd": 1e-5},
    {"hidden": [256, 128, 64],    "lr": 5e-4, "batch": 128, "dropout": 0.2, "wd": 1e-4},
    {"hidden": [64, 32],          "lr": 1e-3, "batch": 256, "dropout": 0.2, "wd": 1e-4},
    {"hidden": [512, 256],        "lr": 5e-4, "batch": 128, "dropout": 0.3, "wd": 1e-4},
]
NN_SEEDS = [0, 1, 2]
NN_MAX_EPOCHS = 100
NN_PATIENCE = 10