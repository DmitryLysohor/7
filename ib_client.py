import logging
import time
from ib_insync import IB, Forex, MarketOrder, LimitOrder, StopOrder, Contract
from config import IB_HOST, IB_PORT, CLIENT_ID

class IBClient:
    def __init__(self, host: str = IB_HOST, port: int = IB_PORT, client_id: int = CLIENT_ID):
        self.ib = IB()
        try:
            self.ib.connect(host, port, clientId=client_id)
            logging.info("Connected to IB Gateway/TWS")
        except Exception as e:
            logging.error(f"Ошибка подключения к IB: {e}")
            raise

        self.eurusd_contract = Forex('EURUSD')
        try:
            self.ib.qualifyContracts(self.eurusd_contract)
        except Exception as e:
            logging.warning(f"Не удалось квалифицировать контракт EURUSD: {e}")

    def get_eurusd_position(self) -> float:
        for pos in self.ib.positions():
            contract = pos.contract
            if isinstance(contract, Contract) \
               and contract.symbol == 'EUR' \
               and contract.secType == 'CASH' \
               and contract.currency == 'USD':
                return pos.position
        return 0.0

    def place_order(self, action: str, quantity: float):
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(self.eurusd_contract, order)
        self.ib.sleep(1)
        status = trade.orderStatus.status
        if status in ('Filled', 'FilledParent'):
            price = trade.orderStatus.avgFillPrice
            logging.info(f"Ордер {action} {quantity} EURUSD исполнен по цене {price}")
        else:
            logging.warning(f"Статус ордера: {status}")
        return trade

    def place_bracket_order(self, side: str, qty: int, sl_price: float, tp_price: float):
        """
        Размещает bracket-ордер: Market → TP (Limit) + SL (Stop)
        """
        parent = MarketOrder(side, qty, transmit=False)

        try:
            self.ib.placeOrder(self.eurusd_contract, parent)

            # Явно ждём появления ордера в системе IB
            for _ in range(20):  # max 2 секунды ожидания
                active_ids = [o.orderId for o in self.ib.openOrders()]
                if parent.orderId in active_ids:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"parent orderId {parent.orderId} not зарегистрирован в IB")

            if side == 'BUY':
                tp_order = LimitOrder('SELL', qty, tp_price, parentId=parent.orderId, tif='GTC', transmit=False)
                sl_order = StopOrder('SELL', qty, sl_price, parentId=parent.orderId, tif='GTC', transmit=True)
            else:
                tp_order = LimitOrder('BUY', qty, tp_price, parentId=parent.orderId, tif='GTC', transmit=False)
                sl_order = StopOrder('BUY', qty, sl_price, parentId=parent.orderId, tif='GTC', transmit=True)

            self.ib.placeOrder(self.eurusd_contract, tp_order)
            self.ib.placeOrder(self.eurusd_contract, sl_order)

            logging.info(
                "BRACKET: %s %d @ Market. SL=%.5f, TP=%.5f (parentId=%s)",
                side, qty, sl_price, tp_price, parent.orderId
            )
        except Exception as e:
            logging.error("BRACKET: не удалось разместить bracket-ордер: %s", e)
            raise

    def disconnect(self):
        self.ib.disconnect()
        logging.info("Disconnected from IB")
