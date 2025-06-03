import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Загружает CSV с колонками: datetime, open, high, low, close (и опционально volume).
    Возвращает DataFrame, отсортированный по возрастанию datetime.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=['datetime', 'open', 'high', 'low', 'close'] + (['volume'] if 'volume' in pd.read_csv(path, nrows=0).columns else []),
        parse_dates=[0],
        usecols=[0, 1, 2, 3, 4] + ([5] if 'volume' in pd.read_csv(path, nrows=0).columns else [])
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
