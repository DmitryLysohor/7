# utils/sessions.py

import pandas as pd

def get_session_label(dt: pd.Timestamp) -> str:
    """
    Определяет торговую сессию по UTC-времени бара.
    Для примера:
      - Asia    = 00:00–08:00 UTC
      - Europe  = 07:00–15:00 UTC
      - US      = 14:30–21:00 UTC
    (В реальности границы можно скорректировать под ваш инструмент.)
    Возвращает строку 'Asia', 'Europe' или 'US', или 'Other'.
    """
    hour = dt.hour
    minute = dt.minute

    if (hour >= 0 and hour < 7) or (hour == 7 and minute < 0):
        return "Asia"
    if (hour >= 7 and hour < 14) or (hour == 14 and minute < 30):
        return "Europe"
    if (hour >= 14 and hour < 21) or (hour == 21 and minute == 0):
        return "US"
    return "Other"

def get_equity_market_phase(dt: pd.Timestamp) -> str:
    """
    Пример для фондового рынка (US). 
    Определим:
      - pre-market      = 04:00–09:30 (UTC-5, но если dt в UTC, надо скорректировать)
      - regular hours   = 09:30–16:00 (UTC-5)
      - after-hours     = 16:00–20:00 (UTC-5)
    Здесь dt – pd.Timestamp в UTC, поэтому нужно вычитать 5 часов:
    """
    # Переведём dt в условное время NY (UTC-5). Для простоты примера:
    ny = dt.tz_localize('UTC').tz_convert('Etc/GMT+5')
    h = ny.hour
    m = ny.minute

    # pre-market
    if (h >= 4 and h < 9) or (h == 9 and m < 30):
        return "pre-market"
    # regular
    if (h == 9 and m >= 30) or (h > 9 and h < 16) or (h == 16 and m == 0):
        return "regular"
    # after-hours
    if (h >= 16 and h < 20):
        return "after-hours"
    return "closed"
