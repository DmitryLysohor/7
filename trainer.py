# trainer.py

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime

from data_utils import load_data                     # Функция загрузки CSV
from pattern_finder import evaluate_patterns, select_best_combo
from shared_utils import build_exact_key, PREFIX_LEN
from config import (
    DATA_FILE,
    LOG_FILE_TRAINER,
    TRAIN_MONTHS, TEST_MONTHS, TOTAL_MONTHS,
    MIN_TOTAL, MIN_RATE,
    TRAINER_OUTPUT_FILE,
    LOOKAHEAD_BARS,
    USE_FIXED_SPREAD, FIXED_SPREAD, SLIPPAGE_PER_TRADE, COMMISSION_PER_TRADE,
    MIN_PATTERN_APPEARANCES
)
from filter_manager import apply_mixed_filter

from filters.time_filter       import TimeFilter
from filters.session_filter    import SessionFilter
from filters.volatility_filter import VolatilityFilter
from filters.mixed_filter      import MixedFilter

from walkforward import find_stable_patterns, find_pattern_occurrences

# ────────────────────────────────────────────────────────────────────────────────
# Логирование
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOG_FILE_TRAINER,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)


def simulate_trade(
    direction: str,
    entry_idx: int,
    cl: float,
    bids: np.ndarray,
    asks: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    datetimes: np.ndarray
) -> dict:
    """
    Общая функция симуляции одной сделки.
    direction: 'BUY' или 'SELL'
    entry_idx: индекс входа (по бару entry_idx мы входим)
    cl: размер TP/SL (float)
    bids, asks, closes, highs, lows, datetimes: numpy-массивы
    Возвращает словарь со следующими полями:
      entry_datetime, entry_price, direction, tp_level, sl_level,
      exit_datetime, exit_price, exit_type, pnl, bars_in_trade
    """
    n = len(closes)

    # Определяем entry_price и spread (игнорируем bids/asks <= 0)
    if (not USE_FIXED_SPREAD
        and bids is not None and asks is not None
        and asks[entry_idx] > 0 and bids[entry_idx] > 0):
        entry_price = asks[entry_idx]
        spread = asks[entry_idx] - bids[entry_idx]
    else:
        entry_price = closes[entry_idx]
        spread = FIXED_SPREAD if USE_FIXED_SPREAD else 0.0

    tp_level = entry_price + cl + spread / 2 + SLIPPAGE_PER_TRADE
    sl_level = entry_price - cl - spread / 2 - SLIPPAGE_PER_TRADE

    # Ищем exit: TP, SL или timeout
    exit_index = None
    exit_type = 'timeout'
    exit_price = None

    start = entry_idx + 1
    end = min(entry_idx + LOOKAHEAD_BARS, n - 1)

    for j in range(start, end + 1):
        if direction == 'BUY':
            if highs[j] >= tp_level:
                exit_index = j
                exit_type = 'TP'
                exit_price = tp_level
                break
            if lows[j] <= sl_level:
                exit_index = j
                exit_type = 'SL'
                exit_price = sl_level
                break
        else:  # SELL
            if lows[j] <= sl_level:
                exit_index = j
                exit_type = 'TP'
                exit_price = sl_level
                break
            if highs[j] >= tp_level:
                exit_index = j
                exit_type = 'SL'
                exit_price = tp_level
                break

    if exit_index is None:
        exit_index = end
        exit_price = closes[exit_index]
        exit_type = 'timeout'

    # Вычисляем PnL
    if exit_type == 'TP':
        pnl = cl - 2 * COMMISSION_PER_TRADE
    elif exit_type == 'SL':
        pnl = -cl - 2 * COMMISSION_PER_TRADE
    else:
        pnl = 0.0

    bars_in_trade = exit_index - entry_idx

    return {
        'entry_datetime': datetimes[entry_idx],
        'entry_price': entry_price,
        'direction': direction,
        'tp_level': tp_level,
        'sl_level': sl_level,
        'exit_datetime': datetimes[exit_index],
        'exit_price': exit_price,
        'exit_type': exit_type,
        'pnl': round(pnl, 6),
        'bars_in_trade': bars_in_trade
    }


def live_backtest(df_live: pd.DataFrame, final_patterns: list[dict], output_csv: str = "live_report.csv") -> pd.DataFrame:
    """
    Выполняет "живой" бэктест на df_live по списку final_patterns.
    Возвращает DataFrame с результатами всех сделок и сохраняет его в CSV.
    """
    trades = []
    last_exit_index_global = -1

    # Заменяем -1 → NaN в колонках bid/ask (если там -1 означает "нет котировок")
    if 'bid' in df_live.columns:
        df_live['bid'] = df_live['bid'].replace(-1, np.nan)
    if 'ask' in df_live.columns:
        df_live['ask'] = df_live['ask'].replace(-1, np.nan)

    # Подготовка массивов
    directions = df_live['direction'].values
    body_sizes = df_live['body_size'].values
    opens      = df_live['open'].values
    highs      = df_live['high'].values
    lows       = df_live['low'].values
    closes     = df_live['close'].values
    bids       = df_live['bid'].values if 'bid' in df_live.columns else None
    asks       = df_live['ask'].values if 'ask' in df_live.columns else None
    datetimes  = df_live['datetime'].values
    n = len(df_live)

    for pat in final_patterns:
        key = pat['key']
        cl = pat['cl']
        w = pat['w']

        # Найдём все вхождения паттерна
        indices = find_pattern_occurrences(key, directions, body_sizes, opens, highs, lows, closes, w)
        for idx in indices:
            entry_bar = idx + 1
            # Пропускаем, если предыдущая сделка ещё не закрылась или выходим за границы
            if entry_bar <= last_exit_index_global or entry_bar >= n:
                continue

            # Определяем направление (BUY/SELL) по первому блоку key
            digits = key.split('-', 1)[0]
            up_cnt = digits.count('1')
            down_cnt = digits.count('0')
            if up_cnt > down_cnt:
                direction = 'BUY'
            elif down_cnt > up_cnt:
                direction = 'SELL'
            else:
                continue  # если равное число '1' и '0', пропускаем

            # Симулируем одну сделку
            trade = simulate_trade(
                direction=direction,
                entry_idx=entry_bar,
                cl=cl,
                bids=bids, asks=asks,
                closes=closes,
                highs=highs, lows=lows,
                datetimes=datetimes
            )
            trades.append(trade)
            last_exit_index_global = np.where(datetimes == trade['exit_datetime'])[0][0]

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(output_csv, index=False)
    return trades_df


def train_and_save_model():
    # 1) Загрузка и проверка данных
    df = load_data(DATA_FILE)
    if df is None or df.empty or len(df) < 200:
        msg = f"Недостаточно данных для тренировки ({0 if df is None else len(df)} строк)."
        print(msg)
        logging.warning(msg)
        return

    # 2) Проверим, что 'datetime' — datetime64, иначе преобразуем
    if 'datetime' not in df.columns or not np.issubdtype(df['datetime'].dtype, np.datetime64):
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=False, errors='raise')
        except Exception as e:
            msg = "Ошибка преобразования 'datetime': " + str(e)
            print(msg)
            logging.error(msg)
            return

    # 3) Препроцессинг (direction, body_size, hour, weekday, month)
    from preprocess import preprocess_df
    df = preprocess_df(df)

    # 4) Отсекаем последние 90 дней для live-теста
    cutoff_live = df['datetime'].max() - pd.Timedelta(days=90)
    df_live = df[df['datetime'] > cutoff_live].reset_index(drop=True)
    df_work = df[df['datetime'] <= cutoff_live].reset_index(drop=True)

    # 5) Walk-forward: ищем "стабильные" паттерны на df_work
    directions = df_work['direction'].values
    body_sizes = df_work['body_size'].values
    opens = df_work['open'].values
    highs = df_work['high'].values
    lows = df_work['low'].values
    closes = df_work['close'].values

    patterns_counts = find_stable_patterns(df_work)
    if not patterns_counts:
        msg = "После walk-forward не найден ни один паттерн."
        print(msg)
        logging.warning(msg)
        return
    logging.info(f"Walk-forward: найдено {len(patterns_counts)} уникальных паттернов.")

    # 6) Фильтрация по MIN_PATTERN_APPEARANCES и Mixed-Filter
    filtered_keys = [k for k, cnt in patterns_counts.items() if cnt >= MIN_PATTERN_APPEARANCES]
    if not filtered_keys:
        msg = f"Нет паттернов с occurrences ≥ {MIN_PATTERN_APPEARANCES}."
        print(msg)
        logging.warning(msg)
        return
    logging.info(f"После MIN_PATTERN_APPEARANCES: {len(filtered_keys)} паттернов осталось.")

    cutoff_for_train = df_work['datetime'].max() - pd.DateOffset(months=TEST_MONTHS)
    train_df = df_work[df_work['datetime'] < cutoff_for_train].reset_index(drop=True)

    final_patterns = apply_mixed_filter(df_work, patterns_counts)
    if not final_patterns:
        msg = "После mixed-фильтрации не осталось паттернов."
        print(msg)
        logging.warning(msg)
        return
    logging.info(f"После mixed-фильтрации осталось {len(final_patterns)} паттернов.")

    # 7) Сохраняем результат в JSON
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "patterns":  final_patterns
    }
    try:
        with open(TRAINER_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        msg = f"Файл {TRAINER_OUTPUT_FILE} сохранён: {len(final_patterns)} паттернов."
        print(msg)
        logging.info(msg)
    except Exception as e:
        msg = f"Ошибка при сохранении {TRAINER_OUTPUT_FILE}: {e}"
        print(msg)
        logging.error(msg)
        return

    # 8) Живой бэктест на последние 90 дней
    trades_df = live_backtest(df_live, final_patterns, output_csv="live_report.csv")
    total = len(trades_df)
    wins = len(trades_df[trades_df["exit_type"] == "TP"])
    win_rate = (wins / total * 100) if total > 0 else 0.0
    print(f"Live Backtest — всего сделок: {total}, WinRate: {win_rate:.2f}%")

if __name__ == "__main__":
    train_and_save_model()
