import numpy as np
import pandas as pd
import logging

from pattern_finder import evaluate_patterns, select_best_combo
from shared_utils import build_exact_key, PREFIX_LEN
from config import (
    TRAIN_MONTHS, TEST_MONTHS, TOTAL_MONTHS,
    MIN_TOTAL, MIN_RATE,
    LOOKAHEAD_BARS
)
from config import USE_FIXED_SPREAD, FIXED_SPREAD, SLIPPAGE_PER_TRADE, COMMISSION_PER_TRADE

def find_pattern_occurrences(
    key: str,
    directions: np.ndarray,
    body_sizes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    window_len: int
) -> list[int]:
    """
    Ищет вхождения паттерна 'key' в массивах directions/body_sizes и возвращает
    список индексов (конец паттерна), при условии, что после idx остаётся хотя бы window_len баров.
    """
    is_ensemble = '|ensemble' in key
    if is_ensemble:
        prefix_key = key.replace('|ensemble', '').strip('-')
        directions_part = prefix_key.split('-', 1)[0]
        pat_len = len(directions_part)
    else:
        directions_part = key.split('-', 1)[0]
        pat_len = len(directions_part)

    n = len(directions)
    max_start = n - pat_len - window_len + 1
    if max_start < 0:
        return []

    matched = []
    for i in range(max_start):
        if is_ensemble:
            prefix_len = pat_len if pat_len < PREFIX_LEN else PREFIX_LEN
            seq_dirs = ''.join(directions[i : i + prefix_len])
            if seq_dirs == prefix_key:
                matched.append(i + pat_len - 1)
        else:
            seq_dirs = ''.join(directions[i : i + pat_len])
            seq_bodies = np.round(body_sizes[i : i + pat_len], 3).tolist()
            cand_key = build_exact_key([
                {'direction': seq_dirs[j], 'body_size': seq_bodies[j]}
                for j in range(pat_len)
            ])
            if cand_key == key:
                matched.append(i + pat_len - 1)

    return matched


def compute_mfe_mae_and_exitbars(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    indices: list[int],
    best_cl: float
) -> tuple[float, float, float, float, float, float, float]:
    """
    Для каждого idx в indices:
      - MFE = max(high[j] - entry_price) j ∈ [idx+1, idx+LOOKAHEAD_BARS]
      - MAE = max(entry_price - low[j])
      - exit_bars = индекс первого hit TP/SL или LOOKAHEAD_BARS
    Возвращает:
      mfe_75, mae_75, median_exit_bars,
      mean_mfe, mean_mae,
      mfe_90, mae_90
    """
    if not indices:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    mfes, maes, exit_bars = [], [], []
    n = len(closes)

    for idx in indices:
        entry_price = closes[idx]
        start = idx + 1
        end = min(idx + LOOKAHEAD_BARS, n - 1)
        if start > end:
            continue

        segment_high = highs[start : end + 1]
        segment_low  = lows[start : end + 1]
        mfe_i = float(np.max(segment_high - entry_price))
        mae_i = float(np.max(entry_price - segment_low))
        mfes.append(mfe_i)
        maes.append(mae_i)

        tp_level = entry_price + best_cl
        sl_level = entry_price - best_cl
        exit_offset = None
        for j in range(start, end + 1):
            if highs[j] >= tp_level:
                exit_offset = j - idx
                break
            if lows[j] <= sl_level:
                exit_offset = j - idx
                break
        if exit_offset is None:
            exit_offset = end - idx
        exit_bars.append(exit_offset)

    if not mfes or not maes or not exit_bars:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    mfe_arr = np.array(mfes)
    mae_arr = np.array(maes)

    mfe_75 = float(np.percentile(mfe_arr, 75))
    mae_75 = float(np.percentile(mae_arr, 75))
    exit_med = float(np.median(exit_bars))
    mean_mfe = float(np.mean(mfe_arr))
    mean_mae = float(np.mean(mae_arr))
    mfe_90 = float(np.percentile(mfe_arr, 90))
    mae_90 = float(np.percentile(mae_arr, 90))

    return mfe_75, mae_75, exit_med, mean_mfe, mean_mae, mfe_90, mae_90


def walk_forward_patterns(
    df: pd.DataFrame,
    directions: np.ndarray,
    body_sizes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray
) -> dict[str, int]:
    """
    Делим df на окна длиной TOTAL_MONTHS, шаг TEST_MONTHS.
    В каждом окне:
      1) train_df (первые TRAIN_MONTHS), test_df (следующие TEST_MONTHS)
      2) best_cl, best_w = select_best_combo(evaluate_patterns(train_df))
      3) evaluate_patterns(test_df).get((best_cl, best_w), []) → отбор по MIN_TOTAL и MIN_RATE
      4) Увеличиваем счётчик для каждого подходящего key
    Возвращает {key: count_windows}
    """
    window_starts = []
    first_date = df['datetime'].min()
    last_date  = df['datetime'].max() - pd.DateOffset(months=TOTAL_MONTHS)

    w = first_date
    while w <= last_date:
        window_starts.append(w)
        w = w + pd.DateOffset(months=TEST_MONTHS)

    all_counts = {}
    total_windows = len(window_starts)

    for idx_w, wstart in enumerate(window_starts):
        pct = int((idx_w + 1) / total_windows * 100)
        print(f"Walk-forward: {pct}% ({idx_w+1}/{total_windows})", end='\r')

        wend = wstart + pd.DateOffset(months=TOTAL_MONTHS)
        window_df = df[(df['datetime'] >= wstart) & (df['datetime'] < wend)].reset_index(drop=True)
        if len(window_df) < 200:
            continue

        cut_train = wstart + pd.DateOffset(months=TRAIN_MONTHS)
        train_df = window_df[window_df['datetime'] < cut_train].reset_index(drop=True)
        test_df  = window_df[window_df['datetime'] >= cut_train].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            continue

        combo_train = evaluate_patterns(train_df)
        best_cl, best_w, _ = select_best_combo(combo_train)

        combo_test = evaluate_patterns(test_df)
        patterns_test = combo_test.get((best_cl, best_w), [])

        for key, rate, total in patterns_test:
            if total < MIN_TOTAL or rate < MIN_RATE:
                continue
            all_counts[key] = all_counts.get(key, 0) + 1

    print(" " * 80, end='\r')
    logging.info(f"Walk-forward окон: {total_windows}, найдено уникальных паттернов: {len(all_counts)}")
    return all_counts


def find_stable_patterns(df: pd.DataFrame) -> dict[str, int]:
    """
    Фасад для вызова walk_forward_patterns:
      - Берёт из df нужные колонки (direction, body_size, open, high, low, close)
      - Возвращает словарь {key: count_windows}
    """
    directions = df['direction'].values
    body_sizes = df['body_size'].values
    opens      = df['open'].values
    highs      = df['high'].values
    lows       = df['low'].values
    closes     = df['close'].values

    return walk_forward_patterns(df, directions, body_sizes, opens, highs, lows, closes)

# ────────────────────────────────────────────────────────────────────────────────
# COMPUTE TRADE STATS (WIN/LOSS/FLAT + PnL)
# ────────────────────────────────────────────────────────────────────────────────
def compute_trade_stats(
    highs: np.ndarray,
    lows: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    closes: np.ndarray,
    indices: list[int],
    best_cl: float
) -> tuple[int, int, int, int, float]:
    """
    Для каждого idx в indices:
      - entry_price = ask[idx] (если есть bid/ask и USE_FIXED_SPREAD=False), иначе close[idx]
      - TP = entry_price + best_cl + spread/2 + slippage
      - SL = entry_price - best_cl - spread/2 + slippage
      - Комиссия учитывается дважды (вход/выход)
    Возвращает: total, wins, losses, flats, total_pnl
    """
    wins = losses = flats = 0
    total_pnl = 0.0
    n = len(closes)

    for idx in indices:
        if not USE_FIXED_SPREAD and bids is not None and asks is not None \
           and not (np.isnan(bids[idx]) or np.isnan(asks[idx])):
            entry_price = asks[idx]
            spread = asks[idx] - bids[idx]
        else:
            entry_price = closes[idx]
            spread = FIXED_SPREAD if USE_FIXED_SPREAD else 0.0

        tp_level = entry_price + best_cl + spread / 2 + SLIPPAGE_PER_TRADE
        sl_level = entry_price - best_cl - spread / 2 - SLIPPAGE_PER_TRADE

        start = idx + 1
        end = min(idx + LOOKAHEAD_BARS, n - 1)
        if start > end:
            losses += 1
            total_pnl -= best_cl
            total_pnl -= 2 * COMMISSION_PER_TRADE
            continue

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
            wins += 1
            total_pnl += best_cl
        elif hit_sl:
            losses += 1
            total_pnl -= best_cl
        else:
            flats += 1  # PnL = 0

        total_pnl -= 2 * COMMISSION_PER_TRADE

    total = len(indices)
    return total, wins, losses, flats, total_pnl
