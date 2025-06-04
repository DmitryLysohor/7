# filters/volatility_filter.py

import pandas as pd
import numpy as np
from typing import Callable, Dict, Any

from filters.base_filter import BaseFilter
from config import LOOKAHEAD_BARS, USE_FIXED_SPREAD, FIXED_SPREAD

class VolatilityFilter(BaseFilter):
    """
    Фильтрация по режимам волатильности (низкая/средняя/высокая) по ATR.
    """

    def __init__(self, atr_period: int, low_th: float, high_th: float, win_rate_th: float, occ_th: int):
        self.atr_period = atr_period
        self.low_th = low_th
        self.high_th = high_th
        self.win_rate_th = win_rate_th
        self.occ_th = occ_th

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Возвращает ATR периода self.atr_period.
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

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
        Разбиваем full_df на 3 режима по ATR:
          ATR < low_th   → «низкая»
          low_th ≤ ATR ≤ high_th → «средняя»
          ATR > high_th → «высокая»
        Для каждой группы считаем вхождения и % выигрышных сделок.
        Если для одного из режимов win_rate ≥ win_rate_th и occurrences ≥ occ_th → True.
        """
        df_stats = full_df.copy().reset_index(drop=True)
        atr_values = self._compute_atr(df_stats)
        df_stats['atr'] = atr_values

        # метки режимов
        df_stats['vol_regime'] = np.where(
            atr_values < self.low_th, 'low',
            np.where(atr_values > self.high_th, 'high', 'medium')
        )

        directions = df_stats['direction'].values
        bodies = df_stats['body_size'].values
        opens = df_stats['open'].values
        highs = df_stats['high'].values
        lows = df_stats['low'].values
        closes = df_stats['close'].values

        bids = df_stats['bid'].values if 'bid' in df_stats.columns else None
        asks = df_stats['ask'].values if 'ask' in df_stats.columns else None

        # разбиение индексов по режимам
        regime_indices = {'low': [], 'medium': [], 'high': []}
        for idx in range(len(df_stats)):
            regime = df_stats.loc[idx, 'vol_regime']
            regime_indices[regime].append(idx)

        # для каждого режима считаем вхождения ключа key
        for regime, idx_list in regime_indices.items():
            if not idx_list:
                continue

            # создадим маски и отфильтруем соответствующие candles
            mask = df_stats.index.isin(idx_list)
            sub_df = df_stats[mask].reset_index(drop=True)

            dirs_sub   = sub_df['direction'].values
            bodies_sub = sub_df['body_size'].values
            opens_sub  = sub_df['open'].values
            highs_sub  = sub_df['high'].values
            lows_sub   = sub_df['low'].values
            closes_sub = sub_df['close'].values

            indices = find_occ_func(key, dirs_sub, bodies_sub, opens_sub, highs_sub, lows_sub, closes_sub, best_w)
            if not indices:
                continue

            wins = 0
            total = 0
            for idx_local in indices:
                if not USE_FIXED_SPREAD and bids is not None and asks is not None and not (np.isnan(bids[idx_local]) or np.isnan(asks[idx_local])):
                    entry_price = asks[idx_local]
                    spread = asks[idx_local] - bids[idx_local]
                else:
                    entry_price = closes_sub[idx_local]
                    spread = FIXED_SPREAD if USE_FIXED_SPREAD else 0.0

                tp = entry_price + best_cl + spread/2
                sl = entry_price - best_cl - spread/2

                start = idx_local + 1
                end = min(idx_local + LOOKAHEAD_BARS, len(closes_sub) - 1)

                hit_tp = False
                hit_sl = False
                for j in range(start, end + 1):
                    if highs_sub[j] >= tp:
                        hit_tp = True
                        break
                    if lows_sub[j] <= sl:
                        hit_sl = True
                        break

                total += 1
                if hit_tp:
                    wins += 1

            if total >= self.occ_th:
                wr = (wins / total) * 100
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
        # Trainer ожидает только time‐статы из TimeFilter, поэтому здесь можно вернуть пустой словарь.
        return {}
