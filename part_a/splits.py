import numpy as np
import pandas as pd

def time_split(df, test_fraction=0.2):
    """Первые (1-test_fraction) — train, последние test_fraction — test.
    Порядок сохраняется как в исходном файле (время наблюдения)."""
    n = len(df)
    n_test = int(round(n * test_fraction))
    train = df.iloc[: n - n_test].reset_index(drop=True)
    test = df.iloc[n - n_test:].reset_index(drop=True)
    return train, test

def expanding_window_folds(train_df, n_folds=3, val_fraction_of_train=0.2):
    """
    Внутри train делаем n_folds последовательных проверок
    с расширяющимся обучающим окном. Валидация всегда ПОСЛЕ обучения.
    Возвращает список (idx_train, idx_val) по позициям в train_df.
    """
    n = len(train_df)
    # делим на (n_folds + 1) блоков примерно равного размера
    block = n // (n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        train_end = block * k
        val_end = block * (k + 1) if k < n_folds else n
        if k == n_folds:
            # последний fold: обучаем на всём до последнего блока,
            # валидация — последний блок
            train_end = n - block
            val_end = n
        idx_tr = np.arange(0, train_end)
        idx_va = np.arange(train_end, val_end)
        if len(idx_va) == 0:
            continue
        folds.append((idx_tr, idx_va))
    return folds