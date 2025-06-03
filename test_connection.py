# test_connection.py

from ib_insync import IB
from config import IB_HOST, IB_PORT, CLIENT_ID

ib = IB()
try:
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    print("Успешно подключились к IB в режиме paper trading")
    # можно, например, спросить текущую серверную дату
    server_time = ib.reqCurrentTime()
    print("Время сервера IB:", server_time)
except Exception as e:
    print("Ошибка подключения:", e)
finally:
    ib.disconnect()
