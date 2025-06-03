import os
import json
import math
import pandas as pd
from datetime import datetime
from signal_generator import generate_signal          # Функция генерации сигналов
from typing import Optional
from config import (
    DATA_FILE,        # Указывает на "EURUSD_iH1.csv"
    TRAINER_OUTPUT_FILE,
)

# ----------------------------------------------------
# Константы для бэктеста
# ----------------------------------------------------
OUTPUT_TRADES_FILE = "backtest_trades.csv"
INITIAL_BALANCE    = 10000.0

# Риск 0.5% капитала на одну сделку (уменьшили с 1% до 0.5%)
RISK_PER_TRADE     = 0.005       

# Базовое проскальзывание (0.5 пипса для EURUSD = 5 пунктов)
BASE_SLIPPAGE      = 0.00005    

# Нижний порог ATR для фильтрации «тихих» часов
# Раньше был 0.0015, теперь уточняем до 0.0012
MIN_ATR_THRESHOLD  = 0.0012     

# Мультипликаторы ATR для TP и SL (как раньше)
TP_ATR_MULTIPLIER    = 1.2
SL_ATR_MULTIPLIER    = 0.8
TRAIL_ATR_MULTIPLIER = 0.5       # для трейлинг-стопа

# Ограничения по объёму
MAX_VOLUME_LOTS = 5.0            # не более 5 лотов в одной сделке

# Комиссия и своп
COMMISSION_PER_TRADE = 5.0       # $5 за сделку
SWAP_PER_BAR         = 0.00001   # $0.00001 за бар


def load_trainer_output(path: str = TRAINER_OUTPUT_FILE, rr_threshold: float = 1.3):
    """
    Читает trainer_output.json и возвращает:
      best_cl, best_w, patterns_list, metrics_dict.

    Фильтруем raw_patterns: оставляем
    только те ключи, у которых RR = avg_mfe/avg_mae ≥ rr_threshold,
    или, если avg_mae == 0, проверяем avg_mfe ≥ rr_threshold * best_cl.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Файл {path} не найден.")
        return None, None, [], {}

    best_cl = data.get("best_cl", 0.0)
    best_w  = data.get("best_w", 0.0)
    raw_patterns = data.get("patterns", [])

    filtered_patterns = []
    metrics = {}
    eps = 1e-9

    for pat in raw_patterns:
        key = pat.get("key")
        avg_mfe = pat.get("avg_mfe", 0.0)
        avg_mae = pat.get("avg_mae", 0.0)
        median_exit = pat.get("median_exit_bars", 1.0)

        # Рассчитываем RR:
        if avg_mae <= 0.0 + eps:
            # Если avg_mae == 0, оставляем только если avg_mfe ≥ rr_threshold * best_cl
            if avg_mfe >= rr_threshold * max(best_cl, eps):
                filtered_patterns.append(key)
                metrics[key] = {
                    "avg_mfe": avg_mfe,
                    "avg_mae": avg_mae,
                    "median_exit_bars": median_exit
                }
        else:
            rr = avg_mfe / avg_mae
            if rr >= rr_threshold:
                filtered_patterns.append(key)
                metrics[key] = {
                    "avg_mfe": avg_mfe,
                    "avg_mae": avg_mae,
                    "median_exit_bars": median_exit
                }

    print(f"После фильтрации осталось {len(filtered_patterns)} паттернов (RR ≥ {rr_threshold}).")
    return best_cl, best_w, filtered_patterns, metrics


def simulate_backtest(
    df: pd.DataFrame,
    best_cl: float,
    best_w:  float,
    patterns_list: list,
    metrics: dict
) -> pd.DataFrame:
    """
    Проход по df (часовые бары) и моделирование вход-выходов с учётом:
      - ATR-основанного TP/SL (TP = entry + TP_ATR_MULTIPLIER*ATR; SL = entry - SL_ATR_MULTIPLIER*ATR)
      - Трейлинг-стопа (TRAIL_ATR_MULTIPLIER * ATR)
      - Риска 0.5% баланса (RISK_PER_TRADE)
      - Динамического проскальзывания (slippage = max(BASE_SLIPPAGE, bar["atr14"] * 0.1))
      - Ограничения объёма (не более MAX_VOLUME_LOTS)
      - Комиссии (COMMISSION_PER_TRADE) и свопа (SWAP_PER_BAR)
      - Торговли только c 08:00 до 17:00 GMT
      - Фильтрации низкой волатильности (bar["atr14"] < MIN_ATR_THRESHOLD)
      - Трендового фильтра по SMA50 (BUY только если close ≥ sma50; SELL только если close ≤ sma50)
    """
    trades = []

    in_position = False
    entry_price = sl_price = tp_price = None
    entry_idx = None
    current_side: Optional[str]    = None
    entry_datetime: Optional[pd.Timestamp] = None

    balance = INITIAL_BALANCE
    idx = 0
    total_bars = len(df)

    while idx < total_bars - 1:
        bar = df.iloc[idx]
        dt  = bar["datetime"]

        # 1) Временной фильтр: торгуем с 08:00 до 17:00 GMT
        hour = dt.hour
        if hour < 8 or hour > 17:
            idx += 1
            continue

        # 2) Фильтр по волатильности: ATR14 должен быть ≥ MIN_ATR_THRESHOLD
        atr14 = bar["atr14"]
        if pd.isna(atr14) or atr14 < MIN_ATR_THRESHOLD:
            idx += 1
            continue

        # 3) Расчёт динамического проскальзывания
        slippage = max(BASE_SLIPPAGE, atr14 * 0.1)

        # 4) Если не в позиции — ищем сигнал
        if not in_position:
            hist_until = df.iloc[: idx + 1].copy()
            signal, matched_key = generate_signal(
                hist_until,
                best_cl,
                best_w,
                patterns_list
            )
            if signal is None or matched_key is None:
                idx += 1
                continue

            # 5) Трендовый фильтр по SMA50:
            close_price = bar["close"]
            sma50 = bar["sma50"]
            if pd.isna(sma50):
                idx += 1
                continue

            if signal == "BUY" and close_price < sma50:
                idx += 1
                continue
            if signal == "SELL" and close_price > sma50:
                idx += 1
                continue

            # 6) Открываем позицию на открытии следующего бара
            next_bar       = df.iloc[idx + 1]
            entry_price    = float(next_bar["open"])
            entry_datetime = next_bar["datetime"]
            entry_atr      = next_bar["atr14"]
            entry_idx      = idx + 1

            # 7) Устанавливаем initial TP/SL, привязанные к ATR
            if signal == "BUY":
                tp_price = entry_price + TP_ATR_MULTIPLIER * entry_atr
                sl_price = entry_price - SL_ATR_MULTIPLIER * entry_atr
                peak_price_for_trail = entry_price
            else:  # SELL
                tp_price = entry_price - TP_ATR_MULTIPLIER * entry_atr
                sl_price = entry_price + SL_ATR_MULTIPLIER * entry_atr
                peak_price_for_trail = entry_price

            # 8) Расчёт объёма (volume) в лотах — правильная формула
            stop_dist = abs(entry_price - sl_price)
            if stop_dist <= 0:
                idx += 1
                continue

            # Не входим, если баланс ≤ 0
            if balance <= 0:
                break

            dollar_risk = balance * RISK_PER_TRADE  # 0.5% текущего
            volume      = dollar_risk / (stop_dist * 100000)
            volume      = max(0.01, round(volume, 2))    # минимум 0.01 лота
            volume      = min(volume, MAX_VOLUME_LOTS)   # максимум MAX_VOLUME_LOTS

            # 9) Если volume < 0.01, пропускаем вход
            if volume < 0.01:
                idx += 1
                continue

            # 10) Входим в позицию
            in_position  = True
            current_side = signal

            # Сразу перескакиваем на entry_idx
            idx = entry_idx
            continue

        # 11) Если мы в позиции — отслеживаем выход
        if in_position:
            # 11.1) Начисляем своп за бар
            balance -= SWAP_PER_BAR * volume * 100000  # 1 лот = 100 000 базовой валюты

            high_j  = bar["high"]
            low_j   = bar["low"]
            atr14_j = bar["atr14"]

            if current_side == "BUY":
                # 11.2) Trailing stop: когда price ≥ entry + 0.5*(TP-entry), подтягиваем SL
                if high_j >= entry_price + 0.5 * (tp_price - entry_price):
                    peak_price_for_trail = max(peak_price_for_trail, high_j)
                    sl_price = max(sl_price, peak_price_for_trail - TRAIL_ATR_MULTIPLIER * atr14_j)

                # 11.3) Проверяем SL/TP внутри бара
                if low_j <= sl_price:
                    exit_price = sl_price - slippage
                    exit_idx   = idx
                elif high_j >= tp_price:
                    exit_price = tp_price - slippage
                    exit_idx   = idx
                else:
                    exit_price = None
                    exit_idx   = None

            else:  # SELL
                # 11.2-b) Trailing stop для SELL
                if low_j <= entry_price - 0.5 * (entry_price - tp_price):
                    peak_price_for_trail = min(peak_price_for_trail, low_j)
                    sl_price = min(sl_price, peak_price_for_trail + TRAIL_ATR_MULTIPLIER * atr14_j)

                # 11.3-b) Проверяем SL/TP внутри бара
                if high_j >= sl_price:
                    exit_price = sl_price + slippage
                    exit_idx   = idx
                elif low_j <= tp_price:
                    exit_price = tp_price + slippage
                    exit_idx   = idx
                else:
                    exit_price = None
                    exit_idx   = None

            # 12) Если достигли SL или TP — фиксируем сделку
            if exit_price is not None:
                pnl = (exit_price - entry_price) * (1 if current_side == "BUY" else -1)
                pnl = pnl * volume * 100000  # P/L в USD
                pnl -= COMMISSION_PER_TRADE
                balance += pnl
                in_position = False

                trades.append({
                    "entry_time":   entry_datetime,
                    "exit_time":    bar["datetime"],
                    "signal":       current_side,
                    "matched_key":  matched_key,
                    "entry_price":  entry_price,
                    "exit_price":   exit_price,
                    "volume":       volume,
                    "pnl":          pnl,
                    "balance":      balance,
                })
                idx += 1
                continue

            # 13) Если не вышли — продолжаем
            idx += 1
            continue

    return pd.DataFrame(trades)


def main():
    # ----------------------------------------
    # Читаем CSV без заголовков (header=None), т.к. первая строка — данные
    # ----------------------------------------
    try:
        df = pd.read_csv(
            DATA_FILE,
            header=None,
            names=["datetime", "open", "high", "low", "close", "volume", "dummy"],
        )
    except FileNotFoundError:
        print(f"[ERROR] Файл {DATA_FILE} не найден.")
        return

    # 1) Преобразуем строку "2009.01.09 17:00" → pd.Timestamp
    try:
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y.%m.%d %H:%M")
    except Exception as e:
        print("[ERROR] Не удалось преобразовать строки в datetime. Проверьте формат:")
        print(str(e))
        return

    # 2) Удаляем столбец "dummy"
    df.drop(columns=["dummy"], inplace=True)

    # 3) Рассчитываем SMA(50)
    df["sma50"] = df["close"].rolling(window=50).mean()

    # 4) Рассчитываем ATR(14) вручную
    df["prev_close"] = df["close"].shift(1)
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = (df["high"] - df["prev_close"]).abs()
    df["tr3"] = (df["low"]  - df["prev_close"]).abs()
    df["tr"]  = df[["tr1", "tr2", "tr3"]].max(axis=1)
    df["atr14"] = df["tr"].rolling(window=14).mean()
    df.drop(columns=["prev_close", "tr1", "tr2", "tr3", "tr"], inplace=True)

    # 5) Убираем первые 49 баров (для SMA) и 13 баров (для ATR14)
    df = df.iloc[max(50, 14):].reset_index(drop=True)

    # 6) Добавляем колонки direction и body_size
    from shared_utils import compute_direction, quantize_body
    df["direction"] = df.apply(lambda r: compute_direction(r["close"], r["open"]), axis=1)
    df["body_size"] = df.apply(lambda r: quantize_body(r["close"], r["open"]), axis=1)

    # 7) Загружаем и фильтруем паттерны (RR ≥ 1.3)
    best_cl, best_w, patterns_list, metrics = load_trainer_output(TRAINER_OUTPUT_FILE, rr_threshold=1.3)
    print(f"best_cl = {best_cl}, best_w = {best_w}")

    if best_cl is None or best_w is None:
        print(f"[ERROR] Не удалось загрузить параметры best_cl/best_w из {TRAINER_OUTPUT_FILE}.")
        return

    # 8) Запуск бэктеста
    trades_df = simulate_backtest(df, best_cl, best_w, patterns_list, metrics)

    # 9) Сохраняем результаты
    trades_df.to_csv(OUTPUT_TRADES_FILE, index=False)
    print(f"Результаты бэктеста сохранены в '{OUTPUT_TRADES_FILE}'.")

    # 10) Выводим общие метрики
    total_trades = len(trades_df)
    win_trades   = trades_df[trades_df["pnl"] > 0.0]
    loss_trades  = trades_df[trades_df["pnl"] <= 0.0]
    total_pnl    = trades_df["pnl"].sum()
    win_rate     = len(win_trades) / total_trades * 100 if total_trades > 0 else 0.0

    print("────────── РЕЗУЛЬТАТЫ БЭКТЕСТА ──────────")
    print(f"Сделок всего:     {total_trades}")
    print(f"Из них профитных: {len(win_trades)}, убыточных: {len(loss_trades)}, Win-rate: {win_rate:.2f}%")
    print(f"Общая P/L (USD):  {total_pnl:.2f}")
    print("────────────────────────────────────────")

    # 11) Рассчитываем доход/убыток по месяцам и годам
    if not trades_df.empty and "exit_time" in trades_df.columns:
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
        # Группировка по месяцу
        trades_df["month"] = trades_df["exit_time"].dt.to_period("M")
        monthly_pl = trades_df.groupby("month")["pnl"].sum().reset_index()
        monthly_pl["month"] = monthly_pl["month"].dt.strftime("%Y-%m")

        # Группировка по году
        trades_df["year"] = trades_df["exit_time"].dt.year
        yearly_pl = trades_df.groupby("year")["pnl"].sum().reset_index()

        print("\nДоход/Убыток по месяцам:")
        for _, row in monthly_pl.iterrows():
            print(f"{row['month']}: {row['pnl']:.2f} USD")

        print("\nДоход/Убыток по годам:")
        for _, row in yearly_pl.iterrows():
            print(f"{row['year']}: {row['pnl']:.2f} USD")


if __name__ == "__main__":
    main()
