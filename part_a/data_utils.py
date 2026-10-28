import hashlib
import pandas as pd
import numpy as np

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_bank_marketing(path):
    df = pd.read_csv(path, sep=";")
    return df

def data_passport(df):
    """Паспорт данных: строки, столбцы, типы, пропуски, спец-значения."""
    info = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": {},
        "unknown_counts": {},
        "special_pdays_999": int((df["pdays"] == 999).sum()),
        "duplicates": int(df.duplicated().sum()),
        "target_distribution": df["y"].value_counts(dropna=False).to_dict(),
    }
    for c in df.columns:
        info["columns"][c] = {
            "dtype": str(df[c].dtype),
            "n_missing": int(df[c].isna().sum()),
            "n_unique": int(df[c].nunique()),
        }
    for c in df.select_dtypes(include="object"):
        info["unknown_counts"][c] = int((df[c] == "unknown").sum())
    return info

def clean(df):
    """
    Правила очистки (сохраняются в журнал):
    - duration удаляется (утечка: известно только после звонка)
    - pdays=999 -> NaN (спец-маркер «не было предыдущего контакта»)
    - 'unknown' оставляем как отдельную категорию (не пропуск)
    - полные дубликаты не удаляем: по заданию совпадение строк ≠ ошибка
    """
    df = df.copy()
    df = df.drop(columns=["duration"])
    df["pdays_was_999"] = (df["pdays"] == 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, np.nan)
    df["y"] = (df["y"] == "yes").astype(int)
    return df

def column_types(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in num_cols if c != "y"]
    return num_cols, cat_cols