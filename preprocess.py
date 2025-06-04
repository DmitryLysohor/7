import numpy as np
import pandas as pd

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет в DataFrame колонки:
      - 'direction' ('1' / '0' / 'neutral')
      - 'body_size' = abs(close - open).round(3)
      - 'hour' (0–23), 'weekday' (0–6), 'month' (1–12)
    """
    df = df.copy()
    df['direction'] = np.where(
        df['close'] > df['open'], '1',
        np.where(df['close'] < df['open'], '0', 'neutral')
    )
    df['body_size'] = (df['close'] - df['open']).abs().round(3)
    df['hour']    = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['month']   = df['datetime'].dt.month
    return df
