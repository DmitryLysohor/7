# utils/indicators.py

import pandas as pd
import numpy as np

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Расчёт ATR (Average True Range) по классической формуле.
    Возвращает Pandas Series той же длины, первые (period-1) значений – NaN.
    """
    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # SMA(или Wilder’s smoothing) для TR
    # Здесь реализуем простую скользящую среднюю для примера
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """
    Простая скользящая средняя.
    """
    return series.rolling(window=period, min_periods=1).mean()

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Вычисляет MACD (fast EMA – slow EMA) и сигнальную линию (EMA от MACD).
    Возвращает DataFrame с колонками ['MACD', 'Signal'].
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "MACD": macd_line,
        "Signal": signal_line
    })

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Расчёт RSI (Relative Strength Index). Возвращает Series.
    """
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ema_up = up.ewm(com=(period - 1), adjust=False).mean()
    ema_down = down.ewm(com=(period - 1), adjust=False).mean()
    rs = ema_up / (ema_down + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(close: pd.Series, period: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками ['BB_middle', 'BB_upper', 'BB_lower'].
      BB_middle = SMA(close, period)
      BB_upper  = middle + mult * std(close, period)
      BB_lower  = middle – mult * std(close, period)
    """
    middle = close.rolling(window=period, min_periods=1).mean()
    std = close.rolling(window=period, min_periods=1).std()
    upper = middle + mult * std
    lower = middle - mult * std
    return pd.DataFrame({
        "BB_middle": middle,
        "BB_upper": upper,
        "BB_lower": lower
    })
