import pytest
from unittest.mock import MagicMock
from ib_insync import MarketOrder, LimitOrder, StopOrder
from ib_client import IBClient  # предполагаем, что ваш класс IBClient в ib_client.py

class DummyOrder:
    def __init__(self, orderId):
        self.orderId = orderId

class DummyOpenOrder:
    def __init__(self, orderId):
        self.orderId = orderId

class DummyIB:
    def __init__(self):
        self.place_order_calls = []
        self._next_order_id = 1001

    def connect(self, host, port, clientId):
        return True  # считаем, что соединение всегда успешно

    def qualifyContracts(self, contract):
        return [contract]

    def placeOrder(self, contract, order):
        # Назначаем orderId, если текущий == 0
        if isinstance(order, MarketOrder) and getattr(order, 'orderId', 0) == 0:
            order.orderId = self._next_order_id
            self._next_order_id += 1
            trade = MagicMock()
            trade.order = order
            trade.orderStatus = MagicMock(status='Submitted', avgFillPrice=1.2345)
            self.place_order_calls.append((contract, order))
            return trade
        else:
            # Обработка child‑ордеров: назначаем orderId, если == 0
            if getattr(order, 'orderId', 0) == 0:
                order.orderId = self._next_order_id
                self._next_order_id += 1
            trade = MagicMock()
            trade.order = order
            trade.orderStatus = MagicMock(status='PreSubmitted', avgFillPrice=0)
            self.place_order_calls.append((contract, order))
            return trade

    def openOrders(self):
        # Возвращаем DummyOpenOrder с родительским orderId=1001
        return [DummyOpenOrder(1001)]

    def sleep(self, seconds):
        pass

@pytest.fixture
def ib_client(monkeypatch):
    # Подменяем __init__ у IBClient, чтобы не подключаться к реальному IB
    def fake_init(self, *args, **kwargs):
        self.ib = DummyIB()
        # Контракт может быть любым, т.к. DummyIB его не проверяет
        self.eurusd_contract = object()

    monkeypatch.setattr(IBClient, '__init__', fake_init)

    return IBClient()

def test_place_bracket_order(ib_client):
    # Параметры для теста
    side = 'BUY'
    qty = 10
    sl_price = 1.2000
    tp_price = 1.3000

    # Вызываем тестируемый метод
    ib_client.place_bracket_order(side, qty, sl_price, tp_price)

    # Проверяем, что было три вызова placeOrder
    calls = ib_client.ib.place_order_calls
    assert len(calls) == 3

    # Первый вызов: родительский MarketOrder
    _, parent_order = calls[0]
    assert isinstance(parent_order, MarketOrder)
    assert parent_order.action == side
    assert parent_order.totalQuantity == qty
    # После вызова DummyIB.placeOrder orderId уже не должен быть 0
    assert parent_order.orderId != 0

    # Второй вызов: Take Profit LimitOrder (SELL)
    _, tp_order = calls[1]
    assert isinstance(tp_order, LimitOrder)
    assert tp_order.action == 'SELL'
    assert tp_order.totalQuantity == qty
    assert pytest.approx(tp_order.lmtPrice) == tp_price
    assert tp_order.parentId == parent_order.orderId

    # Третий вызов: Stop Loss StopOrder (SELL)
    _, sl_order = calls[2]
    assert isinstance(sl_order, StopOrder)
    assert sl_order.action == 'SELL'
    assert sl_order.totalQuantity == qty
    assert pytest.approx(sl_order.auxPrice) == sl_price
    assert sl_order.parentId == parent_order.orderId
