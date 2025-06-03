import numpy as np
import pandas as pd
import logging
from config import (
    MIN_TOTAL, MIN_RATE,
    CHANGE_LIMITS, WINDOWS,
    PATTERN_LEN_START, PATTERN_LEN_END,
    TRAIN_MONTHS, TEST_MONTHS, TOTAL_MONTHS
)
from data_utils import filter_last_n_months

def evaluate_patterns(df_history: pd.DataFrame) -> dict:
    """
    Для каждого сочетания (change_limit, window) возвращает список «хороших» паттернов:
      { (cl, window): [ (pattern_key, rate, total), ... ] }
    где pattern_key — это либо «точный» ключ, либо «префикс-ансамбль».

    Реализовано:
      1) Квантование body_size (BIN_SIZE = 0.002)
      2) Кластеризация по префиксу directions (PREFIX_LEN = 3)
      3) Объединение «точных» и «префиксных» паттернов
      4) **Без вывода прогресса в консоль**
    """
    BIN_SIZE = 0.002
    PREFIX_LEN = 3

    df = df_history.copy()
    df['direction'] = np.where(df['close'] > df['open'], '1',
                               np.where(df['close'] < df['open'], '0', 'neutral'))
    df['body_size'] = (df['close'] - df['open']).abs().round(3)
    df = df[df['direction'] != 'neutral'].reset_index(drop=True)

    data = df.to_dict('records')
    combo_best = {}

    # Внутренняя переменная прогресса убрана
    for cl in CHANGE_LIMITS:
        for window in WINDOWS:
            all_exact = {}
            all_prefix = {}

            for pattern_len in range(PATTERN_LEN_START, PATTERN_LEN_END + 1):
                max_i = len(data) - pattern_len - window
                for i in range(max_i):
                    directions = ''
                    bodies = []
                    valid = True

                    for k in range(pattern_len):
                        candle = data[i + k]
                        if candle['direction'] == 'neutral':
                            valid = False
                            break
                        directions += candle['direction']
                        raw_body = round(candle['body_size'], 3)
                        q = round(raw_body / BIN_SIZE) * BIN_SIZE
                        bodies.append(round(q, 3))
                    if not valid:
                        continue

                    exact_key = directions + '-' + '-'.join(f"{b:.3f}" for b in bodies)

                    up = directions.count('1')
                    down = directions.count('0')
                    if up > down:
                        signal = 'BUY'
                    elif down > up:
                        signal = 'SELL'
                    else:
                        signal = 'HOLD'

                    entry_price = data[i + pattern_len - 1]['close']
                    take_reached = False
                    stop_reached = False

                    for j in range(pattern_len, pattern_len + window):
                        future = data[i + j]
                        if signal == 'BUY':
                            if future['high'] - entry_price >= cl:
                                take_reached = True
                                break
                            if entry_price - future['low'] >= cl:
                                stop_reached = True
                                break
                        elif signal == 'SELL':
                            if entry_price - future['low'] >= cl:
                                take_reached = True
                                break
                            if future['high'] - entry_price >= cl:
                                stop_reached = True
                                break

                    if exact_key not in all_exact:
                        all_exact[exact_key] = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'stop': 0}

                    if signal in ('BUY', 'SELL'):
                        if take_reached:
                            all_exact[exact_key][signal] += 1
                        elif stop_reached:
                            all_exact[exact_key]['stop'] += 1
                        else:
                            all_exact[exact_key]['HOLD'] += 1
                    else:
                        all_exact[exact_key]['HOLD'] += 1

                    effective_prefix_len = min(PREFIX_LEN, len(directions))
                    prefix_directions = directions[:effective_prefix_len]
                    prefix_bodies = bodies[:effective_prefix_len]
                    prefix_key = prefix_directions + '-' + '-'.join(f"{b:.3f}" for b in prefix_bodies)

                    if prefix_key not in all_prefix:
                        all_prefix[prefix_key] = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'stop': 0}

                    if signal in ('BUY', 'SELL'):
                        if take_reached:
                            all_prefix[prefix_key][signal] += 1
                        elif stop_reached:
                            all_prefix[prefix_key]['stop'] += 1
                        else:
                            all_prefix[prefix_key]['HOLD'] += 1
                    else:
                        all_prefix[prefix_key]['HOLD'] += 1

            best_exact = []
            for pattern_key, counts in all_exact.items():
                total = counts['BUY'] + counts['SELL'] + counts['HOLD'] + counts['stop']
                success = counts['BUY'] + counts['SELL']
                if total < MIN_TOTAL or success == 0:
                    continue
                rate = round(100 * success / total, 2)
                if rate >= MIN_RATE:
                    best_exact.append((pattern_key, rate, total))

            best_prefix = []
            for prefix_key, counts in all_prefix.items():
                total = counts['BUY'] + counts['SELL'] + counts['HOLD'] + counts['stop']
                success = counts['BUY'] + counts['SELL']
                if total < MIN_TOTAL or success == 0:
                    continue
                rate = round(100 * success / total, 2)
                if rate >= MIN_RATE:
                    collision = False
                    for exact_key, _, _ in best_exact:
                        if exact_key.startswith(prefix_key):
                            collision = True
                            break
                    if not collision:
                        ensembled_key = f"{prefix_key}|ensemble"
                        best_prefix.append((ensembled_key, rate, total))

            combo_best[(cl, window)] = best_exact + best_prefix

    return combo_best

def select_best_combo(combo_best: dict) -> tuple:
    """
    Из словаря {(cl, w): [...паттерны...]} выбирает пару (cl, w)
    с наибольшим числом «хороших» паттернов.
    Возвращает (best_cl, best_window, patterns_list).
    """
    best_count = -1
    best_cl = None
    best_w = None
    best_list = []

    for (cl, w), patterns in combo_best.items():
        if len(patterns) > best_count:
            best_count = len(patterns)
            best_cl = cl
            best_w = w
            best_list = patterns.copy()

    return best_cl, best_w, best_list
