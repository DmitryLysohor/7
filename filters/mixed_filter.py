# filters/mixed_filter.py

from typing import Dict, Any

from filters.base_filter import BaseFilter
from filters.time_filter import TimeFilter
from filters.session_filter import SessionFilter
from filters.volatility_filter import VolatilityFilter

class MixedFilter(BaseFilter):
    """
    Объединяет несколько под‐фильтров: TimeFilter, SessionFilter, VolatilityFilter.
    Считаем паттерн «годным», если хотя бы один из них возвращает True.
    """

    def __init__(
        self,
        time_filter: TimeFilter,
        session_filter: SessionFilter,
        vol_filter: VolatilityFilter
    ):
        self.time_filter = time_filter
        self.session_filter = session_filter
        self.vol_filter = vol_filter

    def is_good(
        self,
        key: str,
        train_df: "pd.DataFrame",
        full_df: "pd.DataFrame",
        best_cl: float,
        best_w: int,
        find_occ_func: callable,
        compute_trade_stats_func: callable
    ) -> bool:
        """
        Если хотя бы один из трех под‐фильтров считает паттерн «годным» → True.
        """
        if self.time_filter.is_good(
            key,
            train_df,
            full_df,
            best_cl,
            best_w,
            find_occ_func,
            compute_trade_stats_func
        ):
            return True

        if self.session_filter.is_good(
            key,
            train_df,
            full_df,
            best_cl,
            best_w,
            find_occ_func,
            compute_trade_stats_func
        ):
            return True

        if self.vol_filter.is_good(
            key,
            train_df,
            full_df,
            best_cl,
            best_w,
            find_occ_func,
            compute_trade_stats_func
        ):
            return True

        return False

    def collect_stats(
        self,
        key: str,
        df_stats: "pd.DataFrame",
        find_occ_func: callable,
        compute_trade_stats_func: callable,
        cl_best: float,
        w_best: int
    ) -> Dict[str, Any]:
        """
        Возвращаем только статистику от TimeFilter, др. фильтры — требуют дополнительных данных.
        """
        return self.time_filter.collect_stats(
            key,
            df_stats,
            find_occ_func,
            compute_trade_stats_func,
            cl_best,
            w_best
        )
