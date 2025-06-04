# utils/helpers.py

import pandas as pd
import numpy as np

def ensure_datetime(df: pd.DataFrame, col: str = 'datetime') -> pd.DataFrame:
    """
    Гарантирует, что колонка col – это pd.Datetime. 
    Если нет, преобразует через pd.to_datetime.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not in DataFrame")
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def round_price(x: float, precision: int = 5) -> float:
    """
    Округляет цену x до заданного количества знаков после запятой.
    """
    return round(x, precision)

def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    """
    «Сплющивает» вложенные словари:
    {'a': {'x':1, 'y':2}, 'b':3} → {'a_x': 1, 'a_y': 2, 'b':3}
    Удобно, чтобы сохранять «статистику» в табличном виде.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def multi_merge_dfs(dfs: list[pd.DataFrame], on: list[str], how: str = 'left') -> pd.DataFrame:
    """
    Склеивает список DataFrame по столбцам on, как цепочку merge. 
    """
    if not dfs:
        return pd.DataFrame()
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=on, how=how)
    return result
