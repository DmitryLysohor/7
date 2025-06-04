# filters/base_filter.py

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseFilter(ABC):
    """
    Абстрактный класс для «фильтра». Унаследованные фильтры должны реализовать:
      - is_good(...) → bool
      - collect_stats(...) → Dict[str, Any]
    """

    @abstractmethod
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
        Должен вернуть True, если паттерн key считается «годным»:
          либо по своим общим метрикам (overall), либо по специфике (time/session/volatility).
        """

    @abstractmethod
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
        Должен возвращать словарь с необходимыми статистиками.
        Для TimeFilter, например, это {'by_hour': ..., 'by_weekday': ..., 'by_month': ...}.
        """
