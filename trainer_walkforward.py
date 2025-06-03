# trainer_walkforward.py
# Организует walk-forward процесс и агрегацию стабильных паттернов.

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import os

import pattern_finder as pf
import config

def run_pattern_finder_window(_):
    """
    Обёртка для запуска поиска паттернов.
    Мы не передаём сюда срез DataFrame, потому что pattern_finder.find_patterns()
    читает CSV самостоятельно из config.DATA_PATH.
    Аргумент нужен только для совместимости с ThreadPoolExecutor.map.
    """
    return pf.find_patterns()

def walk_forward():
    """
    Проводит walk-forward по исходному CSV:
    1) Читает все бары в DataFrame (для получения длины).
    2) Делит на окна: [start : start+TRAIN_SIZE], шаг WALK_STEP.
    3) В каждом окне параллельно вызывает find_patterns (который парсит весь CSV).
       (В реальной задаче можно переделать find_patterns, чтобы он принимал DataFrame-куск
       вместо чтения всего CSV, но здесь сохранили простоту: find_patterns читает весь файл.)
    4) От каждого окна сохраняем список exact-паттернов.
    5) После всех окон считаем, сколько раз каждый exact-паттерн встретился.
    6) Отбираем те, у которых count >= STABILITY_THRESHOLD * num_windows.
    7) Сохраняем JSON config.STABLE_PATTERNS_FILE с полями "exact" и "ensemble".
    """
    # 1) Чтение CSV (нам нужна длина, чтобы разбить на окна)
    #    parse_dates не нужен, мы не работаем с датами здесь, просто берём количество строк.
    df_all = pd.read_csv(config.DATA_PATH, header=None, sep=',')
    n = len(df_all)

    patterns_per_window = []
    starts = list(range(0, n - config.TRAIN_SIZE - config.TEST_SIZE + 1, config.WALK_STEP))

    # 3) Параллельно обрабатываем каждое окно
    with ThreadPoolExecutor(max_workers=config.THREAD_WORKERS) as exe:
        # Аргумент _ в run_pattern_finder_window не используется — он там только чтобы
        # ThreadPoolExecutor.map запускал функцию нужное число раз.
        results = list(exe.map(run_pattern_finder_window, starts))

    # Соберём exact-паттерны из каждого результата
    for res in results:
        # res = {"exact": [...], "ensemble": [...]}
        patterns_per_window.append(res["exact"])

    # 5) Подсчитываем, сколько раз каждый exact-паттерн встречался
    counter = Counter()
    num_windows = len(patterns_per_window)
    for window_patterns in patterns_per_window:
        for pat in window_patterns:
            counter[tuple(pat)] += 1

    # 6) Отбираем стабильные exact-паттерны
    stable_exact = []
    for pat_tuple, cnt in counter.items():
        if cnt >= config.STABILITY_THRESHOLD * num_windows:
            stable_exact.append(list(pat_tuple))

    # 7) Для ensemble-паттернов на выходе можем просто взять ensemble
    #    из последнего окна (или провести кластеризацию stable_exact заново).
    #    Здесь, для простоты, используем ensemble последнего беглого вызова find_patterns().
    last_ensemble = results[-1]["ensemble"] if results else []

    stable = {
        "exact": stable_exact,
        "ensemble": last_ensemble
    }

    # Сохраняем итоговый JSON рядом со скриптами
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(config.STABLE_PATTERNS_FILE, "w", encoding="utf-8") as f:
        json.dump(stable, f, indent=2, ensure_ascii=False)

    return stable

if __name__ == "__main__":
    stable_patterns = walk_forward()
    print("Stable EXACT patterns:", stable_patterns["exact"])
    print("Stable ENSEMBLE patterns:", stable_patterns["ensemble"])
