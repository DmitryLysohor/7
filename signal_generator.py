# signal_generator.py

import logging
from shared_utils import build_exact_key, build_prefix_key, PREFIX_LEN

def generate_signal(latest_df, best_cl, best_w, patterns_list):
    """
    Проверяет, совпадает ли ключ из последних N свечей (N разный для разных паттернов)
    с одним из паттернов в patterns_list.

    Теперь patterns_list — это просто список строк-ключей.

    Возвращает кортеж (signal, key):
      - signal: 'BUY' или 'SELL', или None (если совпадений нет)
      - key: тот паттерн, который сработал (или None)
    """
    if not patterns_list or best_cl is None:
        return None, None

    # ────────────────────────────────────────────────────────────────────────────
    # 1) Разбиваем паттерны на «точные» (без '|ensemble') и «ensemble» (с '|ensemble')
    # ────────────────────────────────────────────────────────────────────────────
    exact_patterns = []      # список строк-ключей точных паттернов
    ensemble_patterns = {}   # префиксы → True (или любая заглушка)

    for full_key in patterns_list:
        if full_key.endswith("|ensemble"):
            prefix = full_key[:-len("|ensemble")]
            prefix = prefix.rstrip('-')
            ensemble_patterns[prefix] = True
        else:
            exact_patterns.append(full_key)

    # ────────────────────────────────────────────────────────────────────────────
    # 2) Собираем все длины «битовой» части среди exact_patterns
    # ────────────────────────────────────────────────────────────────────────────
    lengths = set(len(key.split("-", 1)[0]) for key in exact_patterns)

    if not lengths and not ensemble_patterns:
        return None, None

    # Сортируем длины по убыванию, чтобы сначала пытаться найти самый длинный паттерн
    sorted_lengths = sorted(lengths, reverse=True)

    # ────────────────────────────────────────────────────────────────────────────
    # ЧАСТЬ A: поиск точного (exact) совпадения
    # ────────────────────────────────────────────────────────────────────────────
    for L in sorted_lengths:
        if len(latest_df) < L:
            continue

        df_tail_records = latest_df.iloc[-L:].to_dict("records")
        exact_key = build_exact_key(df_tail_records)
        if exact_key is None:
            continue

        # Проверяем, есть ли такой ключ среди exact_patterns
        if exact_key in exact_patterns:
            up_count = exact_key.count("1")
            down_count = exact_key.count("0")
            result = "BUY" if up_count > down_count else "SELL"
            logging.info(
                "generate_signal: найден точный паттерн %s (L=%d), up=%d, down=%d → %s",
                exact_key, L, up_count, down_count, result
            )
            return result, exact_key

    # ────────────────────────────────────────────────────────────────────────────
    # ЧАСТЬ B: если точного совпадения нет, ищем ensemble-паттерн (префикс длиной PREFIX_LEN)
    # ────────────────────────────────────────────────────────────────────────────
    if len(latest_df) < PREFIX_LEN:
        return None, None

    df_tail3 = latest_df.iloc[-PREFIX_LEN:].to_dict("records")
    prefix_key = build_prefix_key(df_tail3, prefix_len=PREFIX_LEN)
    if prefix_key is None:
        return None, None

    if prefix_key in ensemble_patterns:
        up3 = prefix_key.count("1")
        down3 = prefix_key.count("0")
        result = "BUY" if up3 > down3 else "SELL"
        full_key = prefix_key + "|ensemble"
        logging.info(
            "generate_signal: найден ensemble-паттерн %s, up3=%d, down3=%d → %s",
            full_key, up3, down3, result
        )
        return result, full_key

    # Ничего не подошло
    logging.info("generate_signal: совпадений нет (ни точных, ни ensemble).")
    return None, None
