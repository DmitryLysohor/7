import json                                        # ← для save_patterns_to_json
import logging                                     # ← для логирования в save_patterns_to_json
import pandas as pd
from datetime import datetime

def load_data(path: str) -> pd.DataFrame:
    """
    Загружает CSV с колонками: datetime, open, high, low, close (и опционально volume).
    Возвращает DataFrame, отсортированный по возрастанию datetime.
    """
    # Сначала проверим, есть ли столбец 'volume' в файле
    sample = pd.read_csv(path, nrows=0)
    has_vol = 'volume' in sample.columns

    usecols = [0, 1, 2, 3, 4] + ([5] if has_vol else [])
    names   = ['datetime', 'open', 'high', 'low', 'close'] + (['volume'] if has_vol else [])

    df = pd.read_csv(
        path,
        header=None,
        names=names,
        parse_dates=[0],
        usecols=usecols
    )
    return df.sort_values('datetime').reset_index(drop=True)


def filter_last_n_months(df: pd.DataFrame, months: int) -> pd.DataFrame:
    """
    Оставляет в df только строки за последние `months` месяцев
    от максимальной даты в df.
    """
    if df.empty:
        return df
    last_dt = df['datetime'].max()
    cutoff = last_dt - pd.DateOffset(months=months)
    return df[df['datetime'] >= cutoff].reset_index(drop=True)


def dropna_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Убирает строки, в которых есть NaN в столбцах open/high/low/close.
    """
    if df[['open', 'high', 'low', 'close']].isnull().any().any():
        return df.dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=True)
    return df


def save_patterns_to_json(filename: str, patterns: list[dict]):
    """
    Сохраняет список паттернов в JSON, добавляя timestamp.
    """
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "patterns":  patterns
    }
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logging.info(f"Файл {filename} сохранён: {len(patterns)} паттернов.")
    except Exception as e:
        logging.error(f"Ошибка при сохранении {filename}: {e}")
