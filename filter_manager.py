import pandas as pd
import numpy as np

from filters.time_filter       import TimeFilter
from filters.session_filter    import SessionFilter
from filters.volatility_filter import VolatilityFilter
from filters.mixed_filter      import MixedFilter

from config import (
    TIME_WIN_RATE_THRESHOLD,
    TIME_OCC_THRESHOLD,
    LOOKAHEAD_BARS,
    USE_FIXED_SPREAD,
    FIXED_SPREAD,
    SLIPPAGE_PER_TRADE,
    COMMISSION_PER_TRADE,
    MIN_RR,
    MAX_MEDIAN_EXIT_BARS,
    MIN_MEAN_MFE,
    MAX_MAE_90,
    TOP_N_COMBOS,
    TEST_MONTHS,
    TOTAL_MONTHS
)

from pattern_finder import evaluate_patterns, select_best_combo
from walkforward import find_pattern_occurrences, compute_mfe_mae_and_exitbars, compute_trade_stats


def apply_mixed_filter(df: pd.DataFrame, patterns_counts: dict) -> list[dict]:
    """
    1) Строим combo_all = evaluate_patterns(train_df)
    2) Берём top_combos по среднему rate
    3) Для каждого ключа из patterns_counts:
         - Подбираем best_cl/best_w по RR
         - overall_good = проверка RR, MFE, MAE, median_exit_bars
         - mixed_filter.is_good(...)
         - Если (overall_good or mixed_good) → сохраняем
    4) Возвращаем список финальных паттернов
    """

    # 1) train_df = df без последних TEST_MONTHS
    cutoff_for_train = df['datetime'].max() - pd.DateOffset(months=TEST_MONTHS)
    train_df = df[df['datetime'] < cutoff_for_train].reset_index(drop=True)

    # 2) Считаем все combos на train_df
    combo_all = evaluate_patterns(train_df)

    # 3) Берём TOP_N_COMBOS по среднему rate
    #    Если для (cl,w) нет паттернов, суммарный rate считается 0
    top_combos = list(sorted(
        ((cl, w) for (cl, w) in combo_all.keys()),
        key=lambda cw: (
            sum(r for _, r, _ in combo_all[cw]) / (len(combo_all[cw]) or 1)
        ),
        reverse=True
    ))[:TOP_N_COMBOS]

    # 4) Полная история для фильтров (df_stats = df)
    df_stats = df.reset_index(drop=True)

    time_filter    = TimeFilter(
        win_rate_th=TIME_WIN_RATE_THRESHOLD,
        occ_th=TIME_OCC_THRESHOLD,
        lookback_bars=len(df_stats)
    )
    session_filter = SessionFilter(win_rate_th=75.0, occ_th=5)
    vol_filter     = VolatilityFilter(
        atr_period=14,
        low_th=0.0005,
        high_th=0.0020,
        win_rate_th=70.0,
        occ_th=5
    )
    mixed_filter = MixedFilter(time_filter, session_filter, vol_filter)

    output = []
    total_keys = len(patterns_counts)

    for idx_k, key in enumerate(patterns_counts):
        pct = int((idx_k + 1) / total_keys * 100)
        print(f"MixedFilter: {pct}% ({idx_k+1}/{total_keys})", end='\r')

        # 5) Подбираем best_cl/w по RR в пределах «overall»-части train_df
        best_rr = -np.inf
        best_candidate = None

        # «overall»-subset: последние TOTAL_MONTHS месяцев train_df
        last_cut = train_df['datetime'].max() - pd.DateOffset(months=TOTAL_MONTHS)
        subset = train_df[train_df['datetime'] >= last_cut].reset_index(drop=True)

        dirs_o   = subset['direction'].values
        bodies_o = subset['body_size'].values
        opens_o  = subset['open'].values
        highs_o  = subset['high'].values
        lows_o   = subset['low'].values
        closes_o = subset['close'].values

        for cl_c, w_c in top_combos:
            indices_o = find_pattern_occurrences(
                key, dirs_o, bodies_o, opens_o, highs_o, lows_o, closes_o, w_c
            )
            if not indices_o:
                continue

            mfe_75, mae_75, med_exit, mean_mfe, mean_mae, mfe_90, mae_90 = \
                compute_mfe_mae_and_exitbars(highs_o, lows_o, closes_o, indices_o, cl_c)

            rr = mfe_75 / mae_75 if mae_75 > 0 else -np.inf

            if rr > best_rr:
                best_rr = rr
                best_candidate = {
                    "cl": float(cl_c),
                    "w":  int(w_c),
                    "mfe_75": round(mfe_75, 6),
                    "mae_75": round(mae_75, 6),
                    "mean_mfe": round(mean_mfe, 6),
                    "mean_mae": round(mean_mae, 6),
                    "mfe_90": round(mfe_90, 6),
                    "mae_90": round(mae_90, 6),
                    "median_exit_bars": round(med_exit, 3),
                    "rr": rr
                }

        if best_candidate is None:
            continue

        # 6) Проверка «общих» порогов (overall_good)
        overall_good = (
            best_candidate["rr"] >= MIN_RR and
            best_candidate["median_exit_bars"] <= MAX_MEDIAN_EXIT_BARS and
            best_candidate["mean_mfe"] >= MIN_MEAN_MFE and
            best_candidate["mae_90"] <= MAX_MAE_90
        )

        # 7) mixed_good = хотя бы один под-фильтр вернул True
        is_mixed_good = mixed_filter.is_good(
            key=key,
            train_df=subset,
            full_df=df,
            best_cl=best_candidate["cl"],
            best_w=best_candidate["w"],
            find_occ_func=find_pattern_occurrences,
            compute_trade_stats_func=compute_trade_stats
        )

        if not (overall_good or is_mixed_good):
            continue

        # 8) Собираем статистику «по времени» (от TimeFilter)
        time_stats = time_filter.collect_stats(
            key,
            df_stats,
            find_pattern_occurrences,
            compute_trade_stats,
            best_candidate["cl"],
            best_candidate["w"]
        )

        pat_dict = {
            "key":               key,
            "cl":                best_candidate["cl"],
            "w":                 best_candidate["w"],
            "avg_mfe":           best_candidate["mfe_75"],
            "avg_mae":           best_candidate["mae_75"],
            "mean_mfe":          best_candidate["mean_mfe"],
            "mean_mae":          best_candidate["mean_mae"],
            "mfe_90":            best_candidate["mfe_90"],
            "mae_90":            best_candidate["mae_90"],
            "median_exit_bars":  best_candidate["median_exit_bars"],
            "count":             int(patterns_counts[key]),
            "tp_multiplier":     round(best_candidate["mfe_75"] / best_candidate["cl"], 3)
                                    if best_candidate["cl"] > 0 else 0.0,
            "sl_multiplier":     round(best_candidate["mae_75"] / best_candidate["cl"], 3)
                                    if best_candidate["cl"] > 0 else 0.0,
            "stats_by_hour":     time_stats["by_hour"],
            "stats_by_weekday":  time_stats["by_weekday"],
            "stats_by_month":    time_stats["by_month"]
        }
        output.append(pat_dict)

    print(" " * 80, end='\r')
    return output
