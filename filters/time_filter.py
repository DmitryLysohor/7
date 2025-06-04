# filters/time_filter.py

import numpy as np
import pandas as pd
from typing import Callable, List, Dict, Any

from filters.base_filter import BaseFilter
from config import LOOKAHEAD_BARS, USE_FIXED_SPREAD, FIXED_SPREAD, SLIPPAGE_PER_TRADE, COMMISSION_PER_TRADE

class TimeFilter(BaseFilter):
    """
    Сбор и проверка статистик по времени (hour/weekday/month).
    """

    def __init__(self, win_rate_th: float, occ_th: int, lookback_bars: int):
        self.win_rate_th = win_rate_th
        self.occ_th = occ_th
        # lookback_bars обычно равен длине всего full_df, но может быть меньше.
        self.lookback_bars = lookback_bars

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
        Проверяем: если по любой «time-бакет» (hour/weekday/month) за последние lookback_bars
        достигается win_rate ≥ win_rate_th и occurrences ≥ occ_th → True.
        """
        # Возьмем последние self.lookback_bars строк full_df (но не меньше):
        if len(full_df) > self.lookback_bars:
            df_stats = full_df.tail(self.lookback_bars).reset_index(drop=True)
        else:
            df_stats = full_df.copy().reset_index(drop=True)

        directions = df_stats['direction'].values
        bodies = df_stats['body_size'].values
        opens = df_stats['open'].values
        highs = df_stats['high'].values
        lows = df_stats['low'].values
        closes = df_stats['close'].values
        hours = df_stats['hour'].values
        weekdays = df_stats['weekday'].values
        months = df_stats['month'].values

        bids = df_stats['bid'].values if 'bid' in df_stats.columns else None
        asks = df_stats['ask'].values if 'ask' in df_stats.columns else None

        # Найдем все вхождения ключа key в этой части истории:
        indices = find_occ_func(key, directions, bodies, opens, highs, lows, closes, best_w)
        if not indices:
            return False

        # Соберем «сырые» бакеты
        sbh = {h: {"total": 0, "wins": 0, "losses": 0, "flats": 0} for h in range(24)}
        sbw = {d: {"total": 0, "wins": 0, "losses": 0, "flats": 0} for d in range(7)}
        sbm = {m: {"total": 0, "wins": 0, "losses": 0, "flats": 0} for m in range(1, 13)}

        for idx_local in indices:
            # Определяем entry_price и spread
            if not USE_FIXED_SPREAD and bids is not None and asks is not None \
               and not (np.isnan(bids[idx_local]) or np.isnan(asks[idx_local])):
                entry_price = asks[idx_local]
                spread = asks[idx_local] - bids[idx_local]
            else:
                entry_price = closes[idx_local]
                spread = FIXED_SPREAD if USE_FIXED_SPREAD else 0.0

            tp = entry_price + best_cl + spread / 2 + SLIPPAGE_PER_TRADE
            sl = entry_price - best_cl - spread / 2 - SLIPPAGE_PER_TRADE

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

            if hit_tp:
                outcome = "win"
            elif hit_sl:
                outcome = "loss"
            else:
                outcome = "flat"

            h = int(hours[idx_local])
            d = int(weekdays[idx_local])
            m = int(months[idx_local])

            sbh[h]["total"] += 1
            sbh[h]["wins"] += 1 if outcome == "win" else 0
            sbh[h]["losses"] += 1 if outcome == "loss" else 0
            sbh[h]["flats"] += 1 if outcome == "flat" else 0

            sbw[d]["total"] += 1
            sbw[d]["wins"] += 1 if outcome == "win" else 0
            sbw[d]["losses"] += 1 if outcome == "loss" else 0
            sbw[d]["flats"] += 1 if outcome == "flat" else 0

            sbm[m]["total"] += 1
            sbm[m]["wins"] += 1 if outcome == "win" else 0
            sbm[m]["losses"] += 1 if outcome == "loss" else 0
            sbm[m]["flats"] += 1 if outcome == "flat" else 0

        # Проверим, есть ли хоть один бакет (hour, weekday или month),
        # у которого win_rate ≥ win_rate_th и occurrences ≥ occ_th
        def check_buckets(raw: Dict[int, Dict[str, int]]) -> bool:
            for v in raw.values():
                t = v["total"]
                if t >= self.occ_th:
                    wr = (v["wins"] / t) * 100
                    if wr >= self.win_rate_th:
                        return True
            return False

        if check_buckets(sbh) or check_buckets(sbw) or check_buckets(sbm):
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
        Возвращает словарь с тремя вложенными словарями:
          {
            'by_hour': {0: {...}, 1: {...}, …, 23: {...}},
            'by_weekday': {0: {...}, …, 6: {...}},
            'by_month': {1: {...}, …, 12: {...}}
          }
        каждая клетка содержит: occurrences, wins, losses, flats, win_rate, avg_pnl.
        Считаем по всей переданной df_stats (обычно — вся история).
        """
        directions = df_stats['direction'].values
        bodies = df_stats['body_size'].values
        opens = df_stats['open'].values
        highs = df_stats['high'].values
        lows = df_stats['low'].values
        closes = df_stats['close'].values
        hours = df_stats['hour'].values
        weekdays = df_stats['weekday'].values
        months = df_stats['month'].values

        bids = df_stats['bid'].values if 'bid' in df_stats.columns else None
        asks = df_stats['ask'].values if 'ask' in df_stats.columns else None

        indices = find_occ_func(key, directions, bodies, opens, highs, lows, closes, w_best)
        if not indices:
            # Заполняем нулями
            return {
                "by_hour": {
                    h: {"occurrences": 0, "wins": 0, "losses": 0, "flats": 0, "win_rate": 0.0, "avg_pnl": 0.0}
                    for h in range(24)
                },
                "by_weekday": {
                    d: {"occurrences": 0, "wins": 0, "losses": 0, "flats": 0, "win_rate": 0.0, "avg_pnl": 0.0}
                    for d in range(7)
                },
                "by_month": {
                    m: {"occurrences": 0, "wins": 0, "losses": 0, "flats": 0, "win_rate": 0.0, "avg_pnl": 0.0}
                    for m in range(1, 13)
                }
            }

        # "сырые" накопители для трех типов группировок
        sbh = {h: {"total": 0, "wins": 0, "losses": 0, "flats": 0, "pnl": 0.0} for h in range(24)}
        sbw = {d: {"total": 0, "wins": 0, "losses": 0, "flats": 0, "pnl": 0.0} for d in range(7)}
        sbm = {m: {"total": 0, "wins": 0, "losses": 0, "flats": 0, "pnl": 0.0} for m in range(1, 13)}

        for idx_local in indices:
            # Определяем entry_price и spread
            if not USE_FIXED_SPREAD and bids is not None and asks is not None \
               and not (np.isnan(bids[idx_local]) or np.isnan(asks[idx_local])):
                entry_price = asks[idx_local]
                spread = asks[idx_local] - bids[idx_local]
            else:
                entry_price = closes[idx_local]
                spread = FIXED_SPREAD if USE_FIXED_SPREAD else 0.0

            tp_level = entry_price + cl_best + spread / 2 + SLIPPAGE_PER_TRADE
            sl_level = entry_price - cl_best - spread / 2 - SLIPPAGE_PER_TRADE

            start = idx_local + 1
            end = min(idx_local + LOOKAHEAD_BARS, len(closes) - 1)

            hit_tp = False
            hit_sl = False
            for j in range(start, end + 1):
                if highs[j] >= tp_level:
                    hit_tp = True
                    break
                if lows[j] <= sl_level:
                    hit_sl = True
                    break

            if hit_tp:
                pnl = cl_best - 2 * COMMISSION_PER_TRADE
                outcome = "win"
            elif hit_sl:
                pnl = -cl_best - 2 * COMMISSION_PER_TRADE
                outcome = "loss"
            else:
                pnl = 0.0
                outcome = "flat"

            h = int(hours[idx_local])
            d = int(weekdays[idx_local])
            m = int(months[idx_local])

            sbh[h]["total"] += 1
            sbh[h]["wins"] += 1 if outcome == "win" else 0
            sbh[h]["losses"] += 1 if outcome == "loss" else 0
            sbh[h]["flats"] += 1 if outcome == "flat" else 0
            sbh[h]["pnl"] += pnl

            sbw[d]["total"] += 1
            sbw[d]["wins"] += 1 if outcome == "win" else 0
            sbw[d]["losses"] += 1 if outcome == "loss" else 0
            sbw[d]["flats"] += 1 if outcome == "flat" else 0
            sbw[d]["pnl"] += pnl

            sbm[m]["total"] += 1
            sbm[m]["wins"] += 1 if outcome == "win" else 0
            sbm[m]["losses"] += 1 if outcome == "loss" else 0
            sbm[m]["flats"] += 1 if outcome == "flat" else 0
            sbm[m]["pnl"] += pnl

        def finalize(raw: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
            out = {}
            for k, v in raw.items():
                t = v["total"]
                if t > 0:
                    wr = round(v["wins"] / t * 100, 2)
                    ap = round(v["pnl"] / t, 6)
                else:
                    wr = 0.0
                    ap = 0.0
                out[k] = {
                    "occurrences": t,
                    "wins":        v["wins"],
                    "losses":      v["losses"],
                    "flats":       v["flats"],
                    "win_rate":    wr,
                    "avg_pnl":     ap
                }
            return out

        return {
            "by_hour": finalize(sbh),
            "by_weekday": finalize(sbw),
            "by_month": finalize(sbm)
        }
