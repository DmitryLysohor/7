# data_fetcher.py

import pandas as pd
import logging
from datetime import datetime, timedelta
from ib_insync import IB, Forex
from config import DATA_FILE

MAX_SECONDS = 86400  # Максимально допустимый интервал в секундах (24 часа)

class DataFetcher:
    """
    Класс для дозаполнения CSV всеми пропущенными часовыми барами EURUSD.
    При инициализации принимает уже готовый IB-инстанс (чтобы не открывать два соединения).
    CSV формат:
      datetime (YYYY.MM.DD HH:MM), open, high, low, close, volume, barCount
    """

    def __init__(self, ib: IB):
        self.ib = ib

    def _get_last_csv_time(self):
        """
        Читает CSV и возвращает pd.Timestamp последнего записанного бара.
        Если CSV не существует или пуст, возвращает None.
        """
        try:
            df = pd.read_csv(
                DATA_FILE,
                header=None,
                names=['datetime', 'open', 'high', 'low', 'close', 'volume', 'barCount'],
                parse_dates=[0],
                usecols=[0]
            )
        except FileNotFoundError:
            return None

        if df.empty:
            return None

        return df['datetime'].max()

    def _bars_to_dataframe(self, bars):
        """
        Преобразует список BarData (IB) в DataFrame pandas с колонками:
        ['datetime', 'open', 'high', 'low', 'close', 'volume', 'barCount'].
        При этом нормализует datetime до naive (без таймзоны).
        """
        records = []
        for bar in bars:
            try:
                bar_dt = pd.to_datetime(bar.date)
            except Exception:
                continue

            # Убираем таймзону, если она есть
            if hasattr(bar_dt, 'tz') and bar_dt.tz is not None:
                bar_dt = bar_dt.tz_convert(None)

            records.append({
                'datetime': bar_dt,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume),
                'barCount': int(bar.barCount)
            })

        if not records:
            return pd.DataFrame(
                columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'barCount']
            )

        df = pd.DataFrame(records)
        # Убедимся, что строки отсортированы по дате (от старых к новым)
        df.sort_values('datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _append_dataframe_to_csv(self, df_chunk):
        """
        Дописывает DataFrame df_chunk в конец CSV. Если CSV не существует, создаёт его.
        Предполагается, что df_chunk уже отсортирован по возрастанию datetime.
        """
        if df_chunk.empty:
            return

        # Преобразуем datetime в строку формата "YYYY.MM.DD HH:MM"
        df_to_write = df_chunk.copy()
        df_to_write['datetime'] = df_to_write['datetime'].dt.strftime('%Y.%m.%d %H:%M')

        # Если файла нет, создаём его и пишем
        try:
            with open(DATA_FILE, 'x', newline='') as f:
                df_to_write.to_csv(f, header=False, index=False)
        except FileExistsError:
            # Файл уже есть — дозаписываем без заголовка
            df_to_write.to_csv(DATA_FILE, mode='a', header=False, index=False)

    def fetch_missing_hour_bars(self):
        """
        Основная функция: смотрит на последний бар в CSV, определяет,
        сколько часов нужно «докупить», разбивает этот интервал на куски ≤24ч,
        запрашивает бары у IB по кускам и дописывает все закрытые бары, которых ещё нет.
        """

        # Шаг 1. Узнаём последний записанный бар
        last_csv_time = self._get_last_csv_time()
        now = datetime.now()

        if last_csv_time is None:
            # Если CSV пуст или отсутствует, считаем, что последний бар был 24 часа назад
            last_csv_time = now - timedelta(seconds=MAX_SECONDS)

        # Если последний бар ≥ now, ничего не делаем
        if last_csv_time >= now:
            logging.info(
                "DataFetcher: Нет пропущенных баров (последний: %s).", last_csv_time
            )
            return

        # Шаг 2. Вычисляем общий пропущенный интервал в секундах
        total_missing_seconds = int((now - last_csv_time).total_seconds())
        # Сколько полных «24-часовых» кусков?
        n_full_chunks = total_missing_seconds // MAX_SECONDS
        # Остаток (меньше 24 часов)
        remainder_seconds = total_missing_seconds % MAX_SECONDS

        # Шаг 3. Строим список (start, end) для каждого куска, начиная от старейшего к новейшему
        chunks = []
        chunk_end = now

        if remainder_seconds > 0:
            chunk_start = chunk_end - timedelta(seconds=remainder_seconds)
            chunks.append((chunk_start, chunk_end))
            chunk_end = chunk_start

        for _ in range(n_full_chunks):
            chunk_start = chunk_end - timedelta(seconds=MAX_SECONDS)
            chunks.append((chunk_start, chunk_end))
            chunk_end = chunk_start

        # Сейчас chunks: [(новейший кусок), …, (самый старый)] — перевернём
        chunks.reverse()

        # Шаг 4. Для каждого куска запросим у IB исторические данные (≤ 86400 сек)
        collected_chunks = []
        contract = Forex('EURUSD')

        for start_dt, end_dt in chunks:
            raw_seconds = int((end_dt - start_dt).total_seconds())

            # Если полный 24‑часовой кусок, запрашиваем ровно 86400 S (без буфера)
            if raw_seconds >= MAX_SECONDS:
                duration_str = f"{MAX_SECONDS} S"
            else:
                # Для неполного куска добавляем +60 сек, чтобы накрыть полностью последний час
                duration_str = f"{raw_seconds + 60} S"

            # endDateTime нужно в формате "YYYYMMDD-HH:MM:SS"
            end_str = end_dt.strftime("%Y%m%d-%H:%M:%S")

            logging.info(
                "DataFetcher: запрашиваем исторические часы EURUSD с %s по %s (duration='%s').",
                start_dt, end_dt, duration_str
            )

            try:
                bars = self.ib.reqHistoricalData(
                    contract=contract,
                    endDateTime=end_str,
                    durationStr=duration_str,
                    barSizeSetting='1 hour',
                    whatToShow='MIDPOINT',
                    useRTH=False,
                    formatDate=1
                )
            except Exception as e:
                logging.error(
                    "DataFetcher: Ошибка reqHistoricalData для %s–%s: %s",
                    start_dt, end_dt, e
                )
                continue

            if not bars:
                logging.warning(
                    "DataFetcher: IB вернул пустой список баров для %s–%s.",
                    start_dt, end_dt
                )
                continue

            df_chunk = self._bars_to_dataframe(bars)
            if df_chunk.empty:
                logging.warning(
                    "DataFetcher: После преобразования – нет строк для %s–%s.",
                    start_dt, end_dt
                )
                continue

            collected_chunks.append(df_chunk)

        if not collected_chunks:
            logging.info("DataFetcher: Новых баров за весь период нет или IB не вернул данные.")
            return

        # Конкатенируем все полученные DataFrame-куски за один раз
        all_new_bars = pd.concat(collected_chunks, ignore_index=True)

        # Шаг 5. Оставляем только «закрытые» бары (datetime < начало текущего часа)
        current_hour_start = now.replace(minute=0, second=0, microsecond=0)
        closed_bars_df = all_new_bars[all_new_bars['datetime'] < pd.Timestamp(current_hour_start)].copy()

        if closed_bars_df.empty:
            logging.info("DataFetcher: Пока нет полностью закрытых часовых баров.")
            return

        # Убираем те, что уже есть в CSV (<= last_csv_time)
        closed_bars_df = closed_bars_df[closed_bars_df['datetime'] > pd.Timestamp(last_csv_time)]
        if closed_bars_df.empty:
            logging.info("DataFetcher: Все закрытые бары уже есть в CSV.")
            return

        # Сортируем и записываем
        closed_bars_df.sort_values('datetime', inplace=True)
        closed_bars_df.reset_index(drop=True, inplace=True)

        self._append_dataframe_to_csv(closed_bars_df)
        logging.info(
            "DataFetcher: Всего дописано %d новых баров.", len(closed_bars_df)
        )
