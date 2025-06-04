# utils/levels.py

import pandas as pd
from typing import List, Tuple

def load_levels_from_csv(path: str) -> List[float]:
    """
    Простая функция: ожидаем CSV, в котором одна колонка «level» – цены поддержки/сопротивления.
    Возвращает список float.
    """
    try:
        df = pd.read_csv(path)
        if 'level' in df.columns:
            return df['level'].dropna().astype(float).tolist()
        else:
            # Если колонка называется иначе, можно подправить
            return df.iloc[:, 0].dropna().astype(float).tolist()
    except Exception:
        return []

def find_nearest_level(price: float, levels: List[float]) -> Tuple[float, float]:
    """
    По цене price находит ближайший уровень из списка levels.
    Возвращает кортеж (nearest_level, расстояние в цене).
    """
    if not levels:
        return (0.0, float('inf'))
    arr = pd.Series(levels)
    idx = (arr - price).abs().idxmin()
    nearest = arr.iloc[idx]
    diff = abs(price - nearest)
    return (nearest, diff)

def is_price_near_any_level(price: float,
                            levels: List[float],
                            tolerance: float = 0.0005) -> bool:
    """
    Возвращает True, если price находится в пределах ±tolerance от любого уровня из списка.
    """
    for lvl in levels:
        if abs(price - lvl) <= tolerance:
            return True
    return False
