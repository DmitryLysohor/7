import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from data_utils import load_data  # Ваша функция загрузки CSV
from pattern_finder import evaluate_patterns, select_best_combo
from config import (
    DATA_FILE,
    TRAIN_MONTHS, TEST_MONTHS, TOTAL_MONTHS,
    MIN_TOTAL, MIN_RATE,
    TRAINER_OUTPUT_FILE,
    LOOKAHEAD_BARS,
    MIN_PATTERN_APPEARANCES,  # Новый параметр
)
from shared_utils import build_exact_key, build_prefix_key, PREFIX_LEN

# Для кластеризации и ML (заглушки)
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(
    filename='trainer.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)


def find_pattern_occurrences(key: str, df: pd.DataFrame, window_len: int) -> list[int]:
    """
    Возвращает список индексов idx, где pattern key встречается,
    с учётом того, что после idx остаётся не менее window_len баров.
    Здесь мы сами рассчитываем 'direction' и 'body_size' для каждой свечи.
    """
    is_ensemble = '|ensemble' in key
    if is_ensemble:
        prefix_key = key.replace('|ensemble', '').strip('-')
        directions_part = prefix_key.split('-', 1)[0]
        pattern_len = len(directions_part)
    else:
        directions_part = key.split('-', 1)[0]
        pattern_len = len(directions_part)

    matched_indices: list[int] = []
    n = len(df)

    # Сразу готовим список словарей с полями 'direction', 'body_size', 'open','high','low','close'
    records: list[dict] = []
    for _, row in df.iterrows():
        if row['close'] > row['open']:
            direction = '1'
        elif row['close'] < row['open']:
            direction = '0'
        else:
            direction = 'neutral'
        body_size = round(abs(row['close'] - row['open']), 3)
        records.append({
            'direction': direction,
            'body_size': body_size,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
        })

    max_start = n - pattern_len - window_len + 1
    if max_start < 0:
        return matched_indices

    for i in range(max_start):
        window_records = records[i : i + pattern_len]
        if is_ensemble:
            # строим префикс из первых prefix_len свечей
            this_prefix_key = build_prefix_key(
                window_records,
                prefix_len=pattern_len if pattern_len < PREFIX_LEN else PREFIX_LEN
            )
            if this_prefix_key == prefix_key:
                matched_indices.append(i + pattern_len - 1)
        else:
            exact_key = build_exact_key(window_records)
            if exact_key == key:
                matched_indices.append(i + pattern_len - 1)

    return matched_indices


def compute_mfe_mae_and_exitbars(df: pd.DataFrame, indices: list[int], best_cl: float) -> tuple[float, float, float]:
    """
    Для каждого idx из indices:
      - вычисляет MFE/MAE за следующие LOOKAHEAD_BARS баров,
      - возвращает 75-й процентиль MFE (avg_mfe), 75-й процентиль MAE (avg_mae) и median_exit_bars.
    """
    high_arr  = df['high'].values
    low_arr   = df['low'].values
    close_arr = df['close'].values

    mfes = []
    maes = []
    exit_bars = []
    n = len(df)

    for idx in indices:
        entry_price = close_arr[idx]
        start = idx + 1
        end   = min(idx + LOOKAHEAD_BARS, n - 1)
        if start > end:
            continue

        segment_high = high_arr[start : end + 1]
        segment_low  = low_arr[start : end + 1]
        mfe_i = float(np.max(segment_high - entry_price))
        mae_i = float(np.max(entry_price - segment_low))
        mfes.append(mfe_i)
        maes.append(mae_i)

        tp_level = entry_price + float(best_cl)
        sl_level = entry_price - float(best_cl)
        exit_bar_offset = None
        for j in range(start, end + 1):
            if high_arr[j] >= tp_level:
                exit_bar_offset = j - idx
                break
            if low_arr[j] <= sl_level:
                exit_bar_offset = j - idx
                break
        if exit_bar_offset is None:
            exit_bar_offset = end - idx
        exit_bars.append(exit_bar_offset)

    if not mfes or not maes or not exit_bars:
        return 0.0, 0.0, 0.0

    mse_75 = float(np.percentile(mfes, 75))
    mae_75 = float(np.percentile(maes, 75))
    median_exit_bars = float(np.median(exit_bars))
    return mse_75, mae_75, median_exit_bars


def cluster_patterns(patterns_list):
    """
    Кластеризация паттернов (заглушка).
    На текущий момент возвращаем исходный список без изменений.
    """
    return patterns_list


def walk_forward_patterns(df: pd.DataFrame) -> dict[str, int]:
    """
    Walk-forward:
    Делим весь df на окна длиной TOTAL_MONTHS (например, 12+3) месяцев
    с шагом TEST_MONTHS (например, 3 мес), собираем паттерны,
    возвращаем словарь {key: count}, где count — сколько раз паттерн прошёл фильтр
    во всех окнах.
    """
    # Шаг 1: собираем все даты старта окон
    window_starts: list[pd.Timestamp] = []
    ws = df['datetime'].min()
    end_date = df['datetime'].max() - pd.DateOffset(months=TOTAL_MONTHS)
    while ws <= end_date:
        window_starts.append(ws)
        ws = ws + pd.DateOffset(months=TEST_MONTHS)

    total_windows = len(window_starts)
    all_patterns_counts: dict[str, int] = {}

    for idx, window_start in enumerate(window_starts):
        pct = int((idx + 1) / total_windows * 100)
        print(f"Walk-forward: {pct}% ({idx+1}/{total_windows} окон)", end='\r')

        window_end = window_start + pd.DateOffset(months=TOTAL_MONTHS)
        window_df = df[(df['datetime'] >= window_start) & (df['datetime'] < window_end)].reset_index(drop=True)
        if len(window_df) < 200:
            continue

        cutoff_train = window_start + pd.DateOffset(months=TRAIN_MONTHS)
        train_df = window_df[window_df['datetime'] < cutoff_train].reset_index(drop=True)
        test_df = window_df[window_df['datetime'] >= cutoff_train].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            continue

        combo_best_train = evaluate_patterns(train_df)
        best_cl, best_w, _ = select_best_combo(combo_best_train)
        combo_best_test = evaluate_patterns(test_df)
        patterns_test = combo_best_test.get((best_cl, best_w), [])

        for key, rate, total in patterns_test:
            if total < MIN_TOTAL or rate < MIN_RATE:
                continue
            all_patterns_counts[key] = all_patterns_counts.get(key, 0) + 1

    print(" " * 80, end='\r')
    logging.info(f"Всего окон в walk-forward: {total_windows}")
    logging.info(f"Счётчики встречаемости паттернов: {all_patterns_counts}")

    return all_patterns_counts


def train_and_save_model():
    df = load_data(DATA_FILE)
    if df.empty or len(df) < 200:
        logging.warning("Недостаточно данных для тренировки.")
        return

    # ------------------------------------------------
    # 1) Walk-forward, получаем словарь {key: count}
    # ------------------------------------------------
    patterns_counts = walk_forward_patterns(df)
    logging.info(f"Walk-forward собрал {len(patterns_counts)} уникальных паттернов с их count")

    # 2) Отбираем только те паттерны, которые встретились ≥ MIN_PATTERN_APPEARANCES
    filtered_keys = [k for k, cnt in patterns_counts.items() if cnt >= MIN_PATTERN_APPEARANCES]
    logging.info(f"Из них {len(filtered_keys)} паттернов встретились ≥ {MIN_PATTERN_APPEARANCES} раз в разных окнах")

    # 3) Берём последние TOTAL_MONTHS для финального тренинга/теста
    df_recent = df[df['datetime'] >= df['datetime'].max() - pd.DateOffset(months=TOTAL_MONTHS)].reset_index(drop=True)
    min_dt = df_recent['datetime'].min()
    cutoff_train = min_dt + pd.DateOffset(months=TRAIN_MONTHS)
    train_df = df_recent[df_recent['datetime'] < cutoff_train].reset_index(drop=True)
    test_df  = df_recent[df_recent['datetime'] >= cutoff_train].reset_index(drop=True)

    combo_best_train = evaluate_patterns(train_df)
    best_cl, best_w, _ = select_best_combo(combo_best_train)
    logging.info(f"Итоговый best_cl={best_cl}, best_w={best_w}")

    # 4) Для каждого ключа из filtered_keys считаем 75%-й перцентиль MFE/MAE и median_exit_bars
    patterns_with_stats: list[dict] = []
    total_patterns = len(filtered_keys)

    for idx, key in enumerate(filtered_keys):
        pct = int((idx + 1) / total_patterns * 100)
        print(f"Финальная фильтрация паттернов: {pct}% ({idx+1}/{total_patterns})", end='\r')

        indices = find_pattern_occurrences(key, test_df, best_w)
        mse_75, mae_75, median_exit_bars = compute_mfe_mae_and_exitbars(test_df, indices, best_cl)
        # Оставляем только те, у кого RR ≥ 1.5 (как было раньше)
        if mae_75 <= 0 or mse_75 / mae_75 < 1.5:
            continue

        # Добавляем в JSON поле "count"
        count = patterns_counts.get(key, 0)

        # Простейшая оценка TP/SL (мультипликаторы относительно best_cl):
        # tp_multiplier = avg_mfe / best_cl, sl_multiplier = avg_mae / best_cl
        # (если best_cl = 0, ставим 0)
        if best_cl > 0:
            tp_mult = round(mse_75 / best_cl, 3)
            sl_mult = round(mae_75 / best_cl, 3)
        else:
            tp_mult = 0.0
            sl_mult = 0.0

        patterns_with_stats.append({
            "key":              key,
            "avg_mfe":          round(mse_75, 6),
            "avg_mae":          round(mae_75, 6),
            "median_exit_bars": round(median_exit_bars, 3),
            "count":            int(count),
            "tp_multiplier":    tp_mult,
            "sl_multiplier":    sl_mult
        })

    print(" " * 80, end='\r')

    # 5) Сохраняем результат в JSON
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "best_cl":   float(best_cl),
        "best_w":    int(best_w),
        "patterns":  patterns_with_stats
    }

    with open(TRAINER_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logging.info(f"Файл {TRAINER_OUTPUT_FILE} сохранён с {len(patterns_with_stats)} паттернами")


if __name__ == "__main__":
    train_and_save_model()
