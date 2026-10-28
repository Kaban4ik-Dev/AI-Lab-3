import hashlib
import numpy as np
from typing import Tuple
import pandas as pd


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_online_retail(path):
    df = pd.read_excel(path, engine="openpyxl")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def data_passport_raw(df):
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": {c: {"dtype": str(df[c].dtype),
                        "n_missing": int(df[c].isna().sum()),
                        "n_unique": int(df[c].nunique())}
                    for c in df.columns},
        "canceled_invoices": int(df["InvoiceNo"].astype(str).str.startswith("C").sum()),
        "nonpositive_quantity": int((df["Quantity"] <= 0).sum()),
        "nonpositive_price": int((df["UnitPrice"] <= 0).sum()),
        "missing_customer": int(df["CustomerID"].isna().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def clean_transactions(df):
    """
    Очистка транзакций:
    - только Quantity > 0 и UnitPrice > 0
    - без отмен (InvoiceNo не начинается с 'C')
    - без пропусков CustomerID
    - без служебных StockCode (без цифр: POST, D, M, BANK CHARGES и т.п.)
    - полные дубликаты не удаляем (по заданию — не доказательство ошибки)
    """
    df = df.copy()
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["CustomerID"].notna()]
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["StockCode"] = df["StockCode"].astype(str)
    df = df[df["StockCode"].str.contains(r"\d", regex=True)]
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_customer_table(df, wholesale_quantile=0.995) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Одна строка = один CustomerID.
    RFM + 2 дополнительных признака + флаг is_wholesale.
    """
    snapshot = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    grp = df.groupby("CustomerID")
    recency = (snapshot - grp["InvoiceDate"].max()).dt.days.rename("Recency")
    frequency = grp["InvoiceNo"].nunique().rename("Frequency")
    monetary = grp["Revenue"].sum().rename("Monetary")
    n_products = grp["StockCode"].nunique().rename("n_products")

    rfm = pd.concat([recency, frequency, monetary, n_products], axis=1).reset_index()
    rfm["avg_basket"] = rfm["Monetary"] / rfm["Frequency"]

    # VIP / опт: экстремальные Monetary ИЛИ Frequency
    m_q = rfm["Monetary"].quantile(wholesale_quantile)
    f_q = rfm["Frequency"].quantile(wholesale_quantile)
    rfm["is_wholesale"] = ((rfm["Monetary"] > m_q) |
                           (rfm["Frequency"] > f_q)).astype(int)
    return rfm, snapshot


def customer_passport(rfm):
    return {
        "n_customers": len(rfm),
        "n_wholesale": int(rfm["is_wholesale"].sum()),
        "columns": {c: {"dtype": str(rfm[c].dtype),
                        "n_missing": int(rfm[c].isna().sum()),
                        "min": float(rfm[c].min()),
                        "max": float(rfm[c].max()),
                        "mean": float(rfm[c].mean()),
                        "median": float(rfm[c].median())}
                    for c in rfm.columns if c != "CustomerID"},
    }