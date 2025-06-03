import os
import json
import math
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from signal_generator import generate_signal          # Функция генерации сигналов
from config import (
    DATA_FILE,        # Обычно "EURUSD_iH1.csv"
    TRAINER_OUTPUT_FILE,
)

# ----------------------------------------------------
# Константы для бэктеста
# ----------------------------------------------------
OUTPUT_TRADES_FILE   = "backtest_trades_detailed.csv"
OUTPUT_PATTERN_STATS = "pattern_stats.csv"
OUTPUT_EQUITY_FILE   = "equity_curve.csv"
INITIAL_BALANCE      = 10000.0

# Риск 0.5% капитала на одну сделку
RISK_PER_TRADE       = 0.005       

# Базовое проскальзывание (≈1 пипс для EURUSD = 10 пунктов → 0.0001)
BASE_SLIPPAGE        = 0.0001    

# Нижний порог ATR для фильтрации «тихих» часов
MIN_ATR_THRESHOLD    = 0.0014     

# Мультипликаторы ATR для TP и SL
TP_ATR_MULTIPLIER    = 1.5
SL_ATR_MULTIPLIER    = 1.0
TRAIL_ATR_MULTIPLIER = 0.5       # для трейлинг-стопа

# Ограничения по объёму
MAX_VOLUME_LOTS      = 2.0       # не более 2 лотов в одной сделке

# Комиссия и своп
COMMISSION_PER_TRADE = 3.0       # $3 за сделку (вход+выход)
SWAP_PER_BAR         = 0.00001   # $0.00001 за каждый бар в позиции

# Параметры динамической фильтрации паттернов
MIN_PATTERN_COUNT        = 5      # минимум 5 сделок по паттерну, чтобы учесть статистику
MIN_AVG_PNL_PER_PATTERN  = 0.0    # отсечем паттерны с avg_pnl_per_tr ≤ 0

def load_previous_pattern_stats(path: str = OUTPUT_PATTERN_STATS):
    """
    Если файл pattern_stats.csv существует, читаем его и возвращаем список
    «хороших» паттернов (count ≥ MIN_PATTERN_COUNT и avg_pnl_per_tr > MIN_AVG_PNL_PER_PATTERN).
    В противном случае возвращаем пустой список.
    """
    if not os.path.isfile(path):
        return []

    df_prev = pd.read_csv(path)
    # Оставляем только те, у которых count ≥ MIN_PATTERN_COUNT и avg_pnl_per_tr > MIN_AVG_PNL_PER_PATTERN
    good = df_prev[
        (df_prev["count"] >= MIN_PATTERN_COUNT) & 
        (df_prev["avg_pnl_per_tr"] > MIN_AVG_PNL_PER_PATTERN)
    ]["pattern_key"].tolist()
    return good


def load_trainer_output(path: str = TRAINER_OUTPUT_FILE, rr_threshold: float = 1.3):
    """
    Читает trainer_output.json и возвращает:
      best_cl, best_w, patterns_list, metrics_dict.

    Фильтруем raw_patterns: оставляем
    только те ключи, у которых RR = avg_mfe/avg_mae ≥ rr_threshold,
    или, если avg_mae == 0, проверяем avg_mfe ≥ rr_threshold * best_cl.

    Затем применяем динамический фильтр:
      - если ранее был файл pattern_stats.csv, оставляем только паттерны из этого «хорошего» списка;
      - иначе оставляем все прошедшие по RR-порогу.
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

    filtered_rr = []
    metrics = {}
    eps = 1e-9

    for pat in raw_patterns:
        key = pat.get("key")
        avg_mfe = pat.get("avg_mfe", 0.0)
        avg_mae = pat.get("avg_mae", 0.0)
        median_exit = pat.get("median_exit_bars", 1.0)

        # Рассчитываем RR:
        if avg_mae <= 0.0 + eps:
            if avg_mfe >= rr_threshold * max(best_cl, eps):
                filtered_rr.append(key)
                metrics[key] = {
                    "avg_mfe": avg_mfe,
                    "avg_mae": avg_mae,
                    "median_exit_bars": median_exit
                }
        else:
            rr = avg_mfe / avg_mae
            if rr >= rr_threshold:
                filtered_rr.append(key)
                metrics[key] = {
                    "avg_mfe": avg_mfe,
                    "avg_mae": avg_mae,
                    "median_exit_bars": median_exit
                }

    print(f"После RR-фильтрации осталось {len(filtered_rr)} паттернов (RR ≥ {rr_threshold}).")

    # Динамический фильтр на основе предыдущих pattern_stats
    good_prev = load_previous_pattern_stats(OUTPUT_PATTERN_STATS)
    if good_prev:
        # Оставляем пересечение фильтра по RR и предыдущих «хороших»
        filtered_patterns = [key for key in filtered_rr if key in good_prev]
        print(f"Динамическая фильтрация: из {len(filtered_rr)} оставлено {len(filtered_patterns)} "
              f"по статистике {OUTPUT_PATTERN_STATS}.")
    else:
        filtered_patterns = filtered_rr
        print("Динамическая фильтрация не применялась (pattern_stats.csv отсутствует).")

    return best_cl, best_w, filtered_patterns, metrics


def simulate_backtest(
    df: pd.DataFrame,
    best_cl: float,
    best_w:  float,
    patterns_list: list,
    metrics: dict
) -> (pd.DataFrame, pd.DataFrame, dict):
    """
    Проход по df (часовые бары) и моделирование вход-выходов с учётом:
      - ATR-основанного TP/SL (TP = entry + TP_ATR_MULTIPLIER*ATR; SL = entry - SL_ATR_MULTIPLIER*ATR)
      - Трейлинг-стопа (TRAIL_ATR_MULTIPLIER * ATR)
      - Риска 0.5% баланса (RISK_PER_TRADE)
      - Динамического проскальзывания (slippage = max(BASE_SLIPPAGE, bar["atr14"] * 0.1))
      - Ограничения объёма (не более MAX_VOLUME_LOTS)
      - Комиссии (COMMISSION_PER_TRADE) и свопа (SWAP_PER_BAR)
      - Торговли только c 09:00 до 16:00 GMT
      - Отключения торговли в Пн/Вт 16:00 GMT
      - Фильтрации низкой волатильности (bar["atr14"] < MIN_ATR_THRESHOLD)
      - Трендового фильтра по SMA50 (BUY только если close ≥ sma50; SELL только если close ≤ sma50)

    Возвращает:
      - trades_df: DataFrame с полным логом сделок
      - equity_df: DataFrame с эквити‐кривой (временная метка, equity)
      - pattern_pnls: dict {pattern_key: [список PnL по каждой сделке]}
    """
    trades = []
    equity_curve = []

    in_position = False
    entry_price = sl_price = tp_price = None
    entry_idx = None
    current_side: Optional[str]    = None
    entry_datetime: Optional[pd.Timestamp] = None
    matched_key: Optional[str]     = None
    volume = 0.0

    balance = INITIAL_BALANCE
    idx = 0
    total_bars = len(df)

    # Словарь для подсчёта статистики паттернов «на лету»
    pattern_pnls = {key: [] for key in patterns_list}

    while idx < total_bars - 1:
        bar = df.iloc[idx]
        dt  = bar["datetime"]

        hour = dt.hour
        dow  = dt.weekday()

        # 1) Ограничение по часам: только с 09:00 до 16:00 GMT
        if hour < 9 or hour > 16:
            idx += 1
            continue

        # 2) Отключаем торговлю в Пн/Вт (dow=0,1) в 16:00 GMT
        if hour == 16 and dow in [0, 1]:
            idx += 1
            continue

        # 3) Фильтр по волатильности: ATR14 должен быть ≥ MIN_ATR_THRESHOLD
        atr14 = bar["atr14"]
        if pd.isna(atr14) or atr14 < MIN_ATR_THRESHOLD:
            idx += 1
            continue

        # 4) Расчёт динамического проскальзывания
        slippage = max(BASE_SLIPPAGE, atr14 * 0.1)

        # 5) Если не в позиции — ищем сигнал
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

            # 6) Трендовый фильтр по SMA50:
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

            # 7) Открываем позицию на открытии следующего бара
            next_bar       = df.iloc[idx + 1]
            entry_price    = float(next_bar["open"])
            entry_datetime = next_bar["datetime"]
            entry_atr      = next_bar["atr14"]
            entry_idx      = idx + 1

            # 8) Устанавливаем initial TP/SL, привязанные к ATR
            if signal == "BUY":
                tp_price = entry_price + TP_ATR_MULTIPLIER * entry_atr
                sl_price = entry_price - SL_ATR_MULTIPLIER * entry_atr
                peak_price_for_trail = entry_price
            else:  # SELL
                tp_price = entry_price - TP_ATR_MULTIPLIER * entry_atr
                sl_price = entry_price + SL_ATR_MULTIPLIER * entry_atr
                peak_price_for_trail = entry_price

            # 9) Расчёт объёма (volume) в лотах
            stop_dist = abs(entry_price - sl_price)
            if stop_dist <= 0 or balance <= 0:
                idx += 1
                continue

            dollar_risk = balance * RISK_PER_TRADE  # 0.5% текущего
            volume      = dollar_risk / (stop_dist * 100000)  # pip_value = 100000 базовой валюты
            volume      = max(0.01, round(volume, 2))    # минимум 0.01 лота
            volume      = min(volume, MAX_VOLUME_LOTS)   # максимум MAX_VOLUME_LOTS

            # 10) Если volume < 0.01, пропускаем вход
            if volume < 0.01:
                idx += 1
                continue

            # 11) Входим в позицию
            in_position  = True
            current_side = signal

            # Переносим idx на бар входа
            idx = entry_idx
            continue

        # 12) Если мы в позиции — отслеживаем выход
        if in_position:
            # 12.1) Начисляем своп за бар
            balance -= SWAP_PER_BAR * volume * 100000  # 1 лот = 100 000 базовой валюты

            bar = df.iloc[idx]
            high_j  = bar["high"]
            low_j   = bar["low"]
            atr14_j = bar["atr14"]

            # 12.2) Trailing stop
            if current_side == "BUY":
                # когда price ≥ entry + 0.5*(TP-entry), подтягиваем SL
                if high_j >= entry_price + 0.5 * (tp_price - entry_price):
                    peak_price_for_trail = max(peak_price_for_trail, high_j)
                    sl_price = max(
                        sl_price,
                        peak_price_for_trail - TRAIL_ATR_MULTIPLIER * atr14_j
                    )

                # Проверяем SL/TP внутри бара
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
                # Trailing stop для SELL
                if low_j <= entry_price - 0.5 * (entry_price - tp_price):
                    peak_price_for_trail = min(peak_price_for_trail, low_j)
                    sl_price = min(
                        sl_price,
                        peak_price_for_trail + TRAIL_ATR_MULTIPLIER * atr14_j
                    )

                # Проверяем SL/TP внутри бара
                if high_j >= sl_price:
                    exit_price = sl_price + slippage
                    exit_idx   = idx
                elif low_j <= tp_price:
                    exit_price = tp_price + slippage
                    exit_idx   = idx
                else:
                    exit_price = None
                    exit_idx   = None

            # 12.3) Если достигли SL или TP — фиксируем сделку
            if exit_price is not None:
                pnl = (exit_price - entry_price) * (1 if current_side == "BUY" else -1)
                pnl = pnl * volume * 100000  # P/L в USD
                pnl -= COMMISSION_PER_TRADE
                balance += pnl
                in_position = False

                trades.append({
                    "entry_time":     entry_datetime,
                    "exit_time":      bar["datetime"],
                    "signal":         current_side,
                    "matched_key":    matched_key,
                    "entry_price":    entry_price,
                    "exit_price":     exit_price,
                    "ATR_on_entry":   entry_atr,
                    "SMA50_on_entry": bar["sma50"] if "sma50" in bar else np.nan,
                    "volume":         volume,
                    "sl_price":       sl_price,
                    "tp_price":       tp_price,
                    "pnl":            pnl,
                    "balance":        balance,
                    "hour_of_day":    bar["datetime"].hour,
                    "day_of_week":    bar["datetime"].weekday(),
                    "commission":     COMMISSION_PER_TRADE,
                    "slippage":       slippage,
                    "swap":           SWAP_PER_BAR * volume * 100000
                })

                # Запишем PnL для статистики по паттерну
                if matched_key in pattern_pnls:
                    pattern_pnls[matched_key].append(pnl)

                # Записываем equity-точку
                equity_curve.append({
                    "time":    bar["datetime"],
                    "equity":  balance
                })

                idx += 1
                continue

            # 12.4) Если не вышли — продолжаем
            idx += 1
            continue

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index("time")

    # 13) Если не было сделок, вернуть «пустые» DataFrame
    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=[
                "entry_time", "exit_time", "signal", "matched_key",
                "entry_price", "exit_price", "ATR_on_entry", "SMA50_on_entry",
                "volume", "sl_price", "tp_price", "pnl", "balance",
                "hour_of_day", "day_of_week", "commission", "slippage", "swap"
            ]
        )
        equity_df = pd.DataFrame(columns=["equity"])

    # 14) Рассчитываем max-drawdown на equity_df
    equity_df = equity_df.sort_index()
    if not equity_df.empty:
        cum_max = equity_df["equity"].cummax()
        drawdown = equity_df["equity"] - cum_max
        max_drawdown_val = drawdown.min()
        max_drawdown_pct = (max_drawdown_val / cum_max.max()) * 100
        print(f"Максимальная просадка: {max_drawdown_val:.2f} USD ({max_drawdown_pct:.2f}%)")
    else:
        print("Эквити-данные отсутствуют — не было ни одной сделки.")

    return trades_df, equity_df, pattern_pnls


def main():
    # ----------------------------------------
    # 1) Читаем CSV (header=None + parse_dates)
    # ----------------------------------------
    try:
        df = pd.read_csv(
            DATA_FILE,
            header=None,
            names=["datetime", "open", "high", "low", "close", "volume", "dummy"],
            parse_dates=[0],
            dayfirst=False
        )
    except FileNotFoundError:
        print(f"[ERROR] Файл {DATA_FILE} не найден.")
        return

    # 2) Рассчитываем SMA(50), ATR(14) вручную
    df["sma50"]      = df["close"].rolling(window=50).mean()
    df["prev_close"] = df["close"].shift(1)
    df["tr1"]        = df["high"] - df["low"]
    df["tr2"]        = (df["high"] - df["prev_close"]).abs()
    df["tr3"]        = (df["low"]  - df["prev_close"]).abs()
    df["tr"]         = df[["tr1", "tr2", "tr3"]].max(axis=1)
    df["atr14"]      = df["tr"].rolling(window=14).mean()

    # Убираем вспомогательные столбцы
    df.drop(columns=["prev_close", "tr1", "tr2", "tr3", "tr"], inplace=True)
    # Сбрасываем первые 50 баров (для SMA50) и 14 баров (для ATR14)
    df = df.iloc[max(50, 14):].reset_index(drop=True)

    # 3) Добавляем колонки direction и body_size
    from shared_utils import compute_direction, quantize_body
    df["direction"] = df.apply(lambda r: compute_direction(r["close"], r["open"]), axis=1)
    df["body_size"] = df.apply(lambda r: quantize_body(r["close"], r["open"]), axis=1)

    # 4) Загружаем и фильтруем паттерны (RR ≥ 1.3 + динамическая фильтрация)
    best_cl, best_w, patterns_list, metrics = load_trainer_output(TRAINER_OUTPUT_FILE, rr_threshold=1.3)
    if best_cl is None or best_w is None:
        print(f"[ERROR] Не удалось загрузить параметры best_cl/best_w из {TRAINER_OUTPUT_FILE}.")
        return

    print(f"best_cl = {best_cl}, best_w = {best_w}")
    print(f"Количество паттернов после фильтрации: {len(patterns_list)}")

    # 5) Запуск бэктеста
    trades_df, equity_df, pattern_pnls = simulate_backtest(df, best_cl, best_w, patterns_list, metrics)

    # 6) Сохраняем результаты сделок и эквити
    trades_df.to_csv(OUTPUT_TRADES_FILE, index=False)
    equity_df.to_csv(OUTPUT_EQUITY_FILE)
    print(f"Расширенные логи сделок сохранены в '{OUTPUT_TRADES_FILE}'.")
    print(f"Эквити-curve сохранена в '{OUTPUT_EQUITY_FILE}'.")

    # 7) Выводим общие метрики
    total_trades = len(trades_df)
    win_trades   = trades_df[trades_df["pnl"] > 0.0]
    loss_trades  = trades_df[trades_df["pnl"] <= 0.0]
    total_pnl    = trades_df["pnl"].sum()
    win_rate     = (len(win_trades) / total_trades * 100) if total_trades > 0 else 0.0

    print("────────── РЕЗУЛЬТАТЫ БЭКТЕСТА ──────────")
    print(f"Сделок всего:     {total_trades}")
    print(f"Из них профитных: {len(win_trades)}, убыточных: {len(loss_trades)}, Win-rate: {win_rate:.2f}%")
    print(f"Общая P/L (USD):  {total_pnl:.2f}")
    print("────────────────────────────────────────")

    # 8) Доход/Убыток по месяцам и годам
    if total_trades > 0:
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
        trades_df["month"]     = trades_df["exit_time"].dt.to_period("M")
        monthly_pl = trades_df.groupby("month")["pnl"].sum().reset_index()
        monthly_pl["month"] = monthly_pl["month"].dt.strftime("%Y-%m")

        trades_df["year"] = trades_df["exit_time"].dt.year
        yearly_pl = trades_df.groupby("year")["pnl"].sum().reset_index()

        print("\nДоход/Убыток по месяцам:")
        for _, row in monthly_pl.iterrows():
            print(f"{row['month']}: {row['pnl']:.2f} USD")

        print("\nДоход/Убыток по годам:")
        for _, row in yearly_pl.iterrows():
            print(f"{row['year']}: {row['pnl']:.2f} USD")

    # 9) Статистика по каждому паттерну
    pattern_stats = []
    for key, pnls in pattern_pnls.items():
        count = len(pnls)
        wins = sum(1 for x in pnls if x > 0)
        losses = count - wins
        total_p = sum(pnls)
        avg_p   = total_p / count if count > 0 else 0.0
        win_rate_key = (wins / count * 100) if count > 0 else 0.0

        pattern_stats.append({
            "pattern_key":    key,
            "count":          count,
            "wins":           wins,
            "losses":         losses,
            "win_rate_%":     round(win_rate_key, 2),
            "total_pnl":      round(total_p, 2),
            "avg_pnl_per_tr": round(avg_p, 2)
        })

    df_pattern_stats = pd.DataFrame(pattern_stats)
    df_pattern_stats = df_pattern_stats.sort_values(by="count", ascending=False)
    df_pattern_stats.to_csv(OUTPUT_PATTERN_STATS, index=False)
    print(f"\nСтатистика по паттернам сохранена в '{OUTPUT_PATTERN_STATS}'.")

    # 10) Рассчитываем и выводим максимальную просадку (дублируем для уверенности)
    if not equity_df.empty:
        equity_df["cum_max"]  = equity_df["equity"].cummax()
        equity_df["drawdown"] = equity_df["equity"] - equity_df["cum_max"]
        max_dd_val = equity_df["drawdown"].min()
        max_dd_pct = (max_dd_val / equity_df["cum_max"].max()) * 100
        print(f"\nМаксимальная просадка (повторно): {max_dd_val:.2f} USD ({max_dd_pct:.2f}%)")

    # Завершение
    print("\nБэктест завершён в полном объёме.")


if __name__ == "__main__":
    main()
