# shared_utils.py

import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Константы, которые должны быть одинаковы в Trainer и в Live Checker
# ────────────────────────────────────────────────────────────────────────────────

BIN_SIZE = 0.002  # шаг квантования тела свечи
PREFIX_LEN = 3    # длина префикса для ensemble-паттернов

# ────────────────────────────────────────────────────────────────────────────────
# 1) Функции для работы с «телом» свечи
# ────────────────────────────────────────────────────────────────────────────────

def quantize_body(close: float, open_: float, bin_size: float = BIN_SIZE) -> float:
    """
    Берёт абсолютную разницу (close - open), округляет до 3 знаков,
    затем квантует по bin_size с «округлением .5 вверх».
    Возвращает float с тремя знаками после запятой.
    """
    raw_body = round(abs(close - open_), 3)
    idx = int(raw_body / bin_size + 0.5)    # «обычное» округление .5 → вверх
    q = idx * bin_size
    return round(q, 3)


def compute_direction(close: float, open_: float) -> str:
    """
    Возвращает '1', если close > open, '0' если close < open.
    Если равны, возвращает 'neutral'.
    """
    if close > open_:
        return "1"
    elif close < open_:
        return "0"
    else:
        return "neutral"


# ────────────────────────────────────────────────────────────────────────────────
# 2) Функции для сборки ключей (точный и префикс)
# ────────────────────────────────────────────────────────────────────────────────

# (оставляем без изменений; в нём точно должны быть функции build_exact_key и build_prefix_key)

def build_exact_key(records: list[dict]) -> str | None:
    """
    Преобразует последовательность bar-dict-ов в строку «directions-body1-body2-…».
    Если встречается нейтральная свеча, возвращает None.
    """
    directions = ''
    bodies = []
    for candle in records:
        if candle.get('direction') == 'neutral':
            return None
        directions += candle['direction']
        bodies.append(candle['body_size'])
    return directions + '-' + '-'.join(f"{b:.3f}" for b in bodies)

def build_prefix_key(records: list[dict], prefix_len: int) -> str | None:
    """
    Строит префикс длины prefix_len по полям 'direction' и 'body_size'.
    Если встречается нейтральная свеча раньше либо нет достаточного числа, возвращает None.
    """
    directions = ''
    bodies = []
    for i in range(prefix_len):
        candle = records[i]
        if candle.get('direction') == 'neutral':
            return None
        directions += candle['direction']
        bodies.append(candle['body_size'])
    return directions + '-' + '-'.join(f"{b:.3f}" for b in bodies)
