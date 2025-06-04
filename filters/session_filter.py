# filters/session_filter.py

import numpy as np
from typing import Callable, Dict, Any
import pandas as pd

from filters.base_filter import BaseFilter
from config import LOOKAHEAD_BARS, USE_FIXED_SPREAD, FIXED_SPREAD

class SessionFilter(BaseFilter):
    """
    Фильтрация по торговым сессиям (Asia, Europe, US). 
    Для примера возьмем простое правило: проверяем, что паттерн хотя бы N раз появлялся в часах 
    конкретной сессии с win_rate ≥ порог.
    """

    def __init__(self, win_rate_th: float, occ_th: int):
        self.win_rate_th = win_rate_th
        self.occ_th = occ_th

        # Определим диапазоны часов (по UTC) для трёх сессий:
        # 👉 0–8: Asia, 8–16: Europe, 16–24: US
        self.sessions = {
            "Asia":   (0, 8),
            "Europe": (8, 16),
            "US":     (16, 24)
        }

    def is_good(
        self,
        key: str,
        train_df: pd.DataFrame,
        full_df: pd.DataFrame,
        best_cl: float,
        best_w: int,
        find_occ_func: Callable,
        compute_trade_stats_func: Callable
    ) -> bool:
        """
        Для каждого диапазона часов считаем: occurrences + wins в этой сессии (за всю full_df).
        Если для какой-то сессии win_rate ≥ win_rate_th и occurrences ≥ occ_th → True.
        """
        df_stats = full_df.copy().reset_index(drop=True)
        directions = df_stats['direction'].values
        bodies = df_stats['body_size'].values
        opens = df_stats['open'].values
        highs = df_stats['high'].values
        lows = df_stats['low'].values
        closes = df_stats['close'].values
        hours = df_stats['hour'].values

        bids = df_stats['bid'].values if 'bid' in df_stats.columns else None
        asks = df_stats['ask'].values if 'ask' in df_stats.columns else None

        # Найдём все вхождения паттерна во всей истории
        indices = find_occ_func(key, directions, bodies, opens, highs, lows, closes, best_w)
        if not indices:
            return False

        # Группируем по сессиям
        session_counts = {
            name: {"total": 0, "wins": 0}
            for name in self.sessions
        }

        for idx_local in indices:
            h = int(hours[idx_local])
            # Определим, к какой сессии относится этот час
            for name, (start_h, end_h) in self.sessions.items():
                if start_h <= h < end_h:
                    # Узнаём результат по этой сделке:
                    if not USE_FIXED_SPREAD and bids is not None and asks is not None and not (np.isnan(bids[idx_local]) or np.isnan(asks[idx_local])):
                        entry_price = asks[idx_local]
                        spread = asks[idx_local] - bids[idx_local]
                    else:
                        entry_price = closes[idx_local]
                        spread = FIXED_SPREAD if USE_FIXED_SPREAD else 0.0

                    tp = entry_price + best_cl + spread/2
                    sl = entry_price - best_cl - spread/2

                    start = idx_local + 1
                    end = min(idx_local + LOOKAHEAD_BARS, len(closes) - 1)

                    hit_tp = False
                    hit_sl = False
                    for j in range(start, end + 1):
                        if highs[j] >= tp:
                            hit_tp = True
                            break
                        if lows[j] <= sl:
                            hit_sl = True
                            break

                    session_counts[name]["total"] += 1
                    if hit_tp:
                        session_counts[name]["wins"] += 1
                    # Проигрыш/flat‐сделки не учитываем как wins

                    break  # нашли сессию, выходим из цикла sessions

        # Проверим каждую сессию
        for cnt in session_counts.values():
            t = cnt["total"]
            if t >= self.occ_th:
                wr = (cnt["wins"] / t) * 100
                if wr >= self.win_rate_th:
                    return True

        return False

    def collect_stats(
        self,
        key: str,
        df_stats: pd.DataFrame,
        find_occ_func: Callable,
        compute_trade_stats_func: Callable,
        cl_best: float,
        w_best: int
    ) -> Dict[str, Any]:
        """
        Здесь можно собрать сколько раз и с каким win_rate паттерн встречался в каждой сессии.
        Но trainer.py ожидает только time‐статы, поэтому вернём пустой словарь.
        """
        return {}
