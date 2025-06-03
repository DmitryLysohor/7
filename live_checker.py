import time
import json
import math
import pandas as pd
import logging
from datetime import datetime, timedelta
from data_utils       import load_data
from signal_generator import generate_signal
from ib_client        import IBClient
from data_fetcher     import DataFetcher
from config           import (
    DATA_FILE,
    LOG_FILE,
    POLL_INTERVAL_MINUTES,
    TRAINER_OUTPUT_FILE
)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def load_trainer_output(path: str = TRAINER_OUTPUT_FILE):
    """
    Читает trainer_output.json и возвращает:
      best_cl, best_w, patterns_list, metrics_dict

    Где:
      - patterns_list: [ (key, rate, total), ... ] 
      - metrics_dict[key] = {'avg_mfe': float, 'avg_mae': float, 'avg_exit_bars': float}
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error("Файл %s не найден.", path)
        return None, None, [], {}

    best_cl = data.get("best_cl")
    best_w  = data.get("best_w")
    raw_patterns = data.get("patterns", [])

    patterns_list = []
    metrics = {}
    for pat in raw_patterns:
        raw_key = pat["key"]
        rate    = pat["rate"]
        total   = pat["total"]

        patterns_list.append((raw_key, rate, total))
        metrics[raw_key] = {
            "avg_mfe":        pat.get("avg_mfe", 0.0),
            "avg_mae":        pat.get("avg_mae", 0.0),
            "avg_exit_bars":  pat.get("avg_exit_bars", 0.0)
        }

    return best_cl, best_w, patterns_list, metrics


def sleep_until_next_interval(interval_minutes: int):
    """
    Спим до ближайшего будущего времени, кратного interval_minutes.
    Например, если interval_minutes=5 и сейчас 13:02:17, то уснёт на 2 мин 43 сек,
    чтобы проснуться в 13:05:00.
    """
    now = datetime.now()
    minute = now.minute
    second = now.second
    micro = now.microsecond

    remainder = minute % interval_minutes
    if remainder == 0 and second == 0 and micro == 0:
        wait_seconds = 0
    else:
        minutes_to_add = (interval_minutes - remainder) if remainder != 0 else interval_minutes
        next_time = (now.replace(second=0, microsecond=0)
                     + timedelta(minutes=minutes_to_add))
        delta = next_time - now
        wait_seconds = delta.total_seconds()

    if wait_seconds > 0:
        logging.info(
            "Ждём %.0f секунд до следующего запуска (в %s).",
            wait_seconds,
            (now + timedelta(seconds=wait_seconds)).strftime("%H:%M:%S")
        )
        time.sleep(wait_seconds)


def main_loop():
    try:
        ib_client = IBClient()
        data_fetcher = DataFetcher(ib_client.ib)
    except Exception as e:
        logging.error("Ошибка инициализации Live Checker: %s", e)
        return

    # ─────────────── Состояние открытой позиции ───────────────
    in_position = False
    entry_datetime = None
    exit_timeout = None
    current_side = None

    QTY = 10000  # количество контрактов (константа)

    first_run = True

    while True:
        try:
            # ───────────────────────────────────────────────────
            # 1) Загружаем модель (best_cl, best_w, patterns_list, metrics)
            # ───────────────────────────────────────────────────
            best_cl, best_w, best_patterns, metrics = load_trainer_output()
            logging.info("Загруженные паттерны: %s", best_patterns)
            if best_cl is None or not best_patterns:
                logging.warning("Нет обученной модели или пустой список паттернов → пропускаем.")
                sleep_until_next_interval(POLL_INTERVAL_MINUTES)
                continue

            # ───────────────────────────────────────────────────
            # 2) Дозаполняем CSV (если у вас так настроено)
            # ───────────────────────────────────────────────────
            data_fetcher.fetch_missing_hour_bars()

            # ───────────────────────────────────────────────────
            # 3) Загружаем полный DataFrame (включая свежие бары)
            # ───────────────────────────────────────────────────
            full_df = load_data(DATA_FILE)
            if full_df.empty:
                logging.warning("CSV пуст после дозагрузки.")
                sleep_until_next_interval(POLL_INTERVAL_MINUTES)
                continue

            # ───────────────────────────────────────────────────
            # 4) Проверяем, не пора ли выходить по таймауту (если in_position=True)
            # ───────────────────────────────────────────────────
            # Узнаём текущую позицию у брокера
            current_pos = ib_client.get_eurusd_position()
            logging.info("Текущая позиция EURUSD: %s", current_pos)

            if in_position and current_pos == 0:
                # Позиция была закрыта брокером (SL или TP сработало)
                logging.info(
                    "Позиция (%s) закрылась брокером по SL/TP. Сброс состояния.",
                    current_side
                )
                in_position = False
                entry_datetime = None
                exit_timeout = None
                current_side = None

            elif in_position and current_pos != 0:
                # Мы всё ещё в позиции: смотрим, сколько баров прошло
                # находим индекс entry_datetime в full_df
                try:
                    idx_entry = full_df[full_df["datetime"] == entry_datetime].index[0]
                    idx_latest = len(full_df) - 1
                    bars_held = idx_latest - idx_entry
                except IndexError:
                    # Иногда entry_datetime может не совпасть точно — логируем и пропускаем проверку
                    bars_held = 0
                    logging.warning("Не удалось найти точное entry_datetime %s в full_df.", entry_datetime)

                if exit_timeout is not None and bars_held >= exit_timeout:
                    # По таймауту принудительно закрываем позицию по рынку
                    if current_side == "BUY":
                        # Чтобы закрыть BUY, нужно SELL QTY по рынку
                        ib_client.place_bracket_order("SELL", QTY, 0, 0)
                        logging.info(
                            "TIMEOUT: закрываем BUY по рынку через %d баров (entry=%s).",
                            bars_held, entry_datetime
                        )
                    else:  # current_side == "SELL"
                        ib_client.place_bracket_order("BUY", QTY, 0, 0)
                        logging.info(
                            "TIMEOUT: закрываем SELL по рынку через %d баров (entry=%s).",
                            bars_held, entry_datetime
                        )

                    # Сбрасываем состояние
                    in_position = False
                    entry_datetime = None
                    exit_timeout = None
                    current_side = None

            # ───────────────────────────────────────────────────
            # 5) Генерация сигнала по последнему бару (H1)
            # ───────────────────────────────────────────────────
            signal, matched_key = generate_signal(full_df, best_cl, best_w, best_patterns)

            if signal is None or matched_key is None:
                logging.info("Никаких совпадений — продолжаем ждать.")
            else:
                # Если мы уже в позиции, не открываем новую (ожидаем закрытия по SL/TP или TIMEOUT)
                if in_position and ib_client.get_eurusd_position() != 0:
                    logging.info(
                        "Сигнал %s по ключу %s пришёл, но мы всё ещё в позиции (%s). Пропускаем.",
                        signal, matched_key, current_side
                    )
                    # пропускаем открытие новой позиции
                    pass
                else:
                    # ───────────────────────────────────────────────
                    # 6) Вход в новую позицию
                    # ───────────────────────────────────────────────
                    # Узнаём SL/TP/EXIT для matched_key
                    metric = metrics.get(matched_key, {
                        "avg_mfe": 0.0,
                        "avg_mae": 0.0,
                        "avg_exit_bars": 0.0
                    })
                    avg_mfe       = metric["avg_mfe"]
                    avg_mae       = metric["avg_mae"]
                    avg_exit_bars = metric["avg_exit_bars"]

                    if avg_mfe <= 0.0:
                        avg_mfe = best_cl
                    if avg_mae <= 0.0:
                        avg_mae = best_cl

                    # Округляем таймаут вверх минимум в 1 бар
                    exit_timeout = max(1, math.ceil(avg_exit_bars))

                    # Цена последнего бара (close) и время
                    entry_price = float(full_df['close'].iloc[-1])
                    entry_datetime = full_df['datetime'].iloc[-1]

                    # Вычисляем уровни SL/TP
                    if signal == "BUY":
                        sl_price = entry_price - avg_mae
                        tp_price = entry_price + avg_mfe
                    else:  # "SELL"
                        sl_price = entry_price + avg_mae
                        tp_price = entry_price - avg_mfe

                    # Открываем новую позицию с бракет-ордерами
                    if signal == "BUY":
                        ib_client.place_bracket_order("BUY", QTY, sl_price, tp_price)
                        current_side = "BUY"
                    else:  # "SELL"
                        ib_client.place_bracket_order("SELL", QTY, sl_price, tp_price)
                        current_side = "SELL"

                    in_position = True

                    logging.info(
                        "BRACKET: открыт %s %d по рынку, SL=%.5f, TP=%.5f, exit_timeout=%d баров (key=%s)",
                        signal, QTY, sl_price, tp_price, exit_timeout, matched_key
                    )

        except Exception as ex:
            logging.error("Исключение в main_loop: %s", ex)

        # ───────────────────────────────────────────────────
        # 7) Ждём до следующего «круглого» интервала
        # ───────────────────────────────────────────────────
        if first_run:
            first_run = False
            sleep_until_next_interval(POLL_INTERVAL_MINUTES)
        else:
            sleep_until_next_interval(POLL_INTERVAL_MINUTES)

    ib_client.disconnect()


if __name__ == "__main__":
    main_loop()
