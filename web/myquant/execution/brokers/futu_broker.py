"""
Futu OpenAPI Broker — for HK, US and A-share trading via Futu/MooMoo.

Requires:
    pip install futu-api
    FutuOpenD desktop app running on FUTU_HOST:FUTU_PORT
"""
from __future__ import annotations

from myquant.config.logging_config import get_logger
from myquant.config.settings import settings
from myquant.execution.brokers.base_broker import BaseBroker
from myquant.models.order import Order, OrderSide, OrderStatus, OrderType

logger = get_logger(__name__)


class FutuBroker(BaseBroker):
    """
    Futu OpenAPI broker implementation.

    Handles:
        - HK stocks (hk prefix)
        - US stocks (us prefix)
        - A-shares via Futu (sh/sz prefix)
    """

    def __init__(self) -> None:
        super().__init__()
        self._trade_ctx = None
        self._quote_ctx = None
        self._account_id: str = ""

    # ── Lifecycle ─────────────────────────────────────────────

    async def connect(self) -> None:
        try:
            import futu as ft  # type: ignore[import]

            self._quote_ctx = ft.OpenQuoteContext(
                host=settings.FUTU_HOST, port=settings.FUTU_PORT
            )
            self._trade_ctx = ft.OpenSecTradeContext(
                filter_trdmarket=ft.TrdMarket.HK,
                host=settings.FUTU_HOST,
                port=settings.FUTU_PORT,
                security_firm=ft.SecurityFirm.FUTUSECURITIES,
            )

            ret, data = self._trade_ctx.unlock_trade(
                password_md5=settings.FUTU_TRADE_PASSWORD_MD5
            )
            if ret != ft.RET_OK:
                raise ConnectionError(f"Futu trade unlock failed: {data}")

            # Get account info
            ret, acc_list = self._trade_ctx.get_acc_list()
            if ret == ft.RET_OK and not acc_list.empty:
                self._account_id = str(acc_list.iloc[0]["acc_id"])

            logger.info(
                "FutuBroker connected. Account: %s", self._account_id
            )
        except ImportError:
            raise RuntimeError(
                "futu-api not installed. Run: pip install futu-api"
            )

    async def disconnect(self) -> None:
        if self._trade_ctx:
            self._trade_ctx.close()
        if self._quote_ctx:
            self._quote_ctx.close()
        logger.info("FutuBroker disconnected.")

    # ── Trading ───────────────────────────────────────────────

    async def submit_order(self, order: Order) -> str:
        import futu as ft  # type: ignore[import]

        code = self._to_futu_code(order.symbol)
        trd_side = ft.TrdSide.BUY if order.side == OrderSide.BUY else ft.TrdSide.SELL
        order_type = (
            ft.OrderType.MARKET
            if order.order_type == OrderType.MARKET
            else ft.OrderType.NORMAL
        )

        ret, data = self._trade_ctx.place_order(
            price=order.limit_price or 0,
            qty=order.quantity,
            code=code,
            trd_side=trd_side,
            order_type=order_type,
            trd_env=ft.TrdEnv.REAL if settings.IS_LIVE else ft.TrdEnv.SIMULATE,
            acc_id=int(self._account_id),
        )

        if ret != ft.RET_OK:
            order.status = OrderStatus.REJECTED
            order.notes = str(data)
            self._notify_reject(order, str(data))
            raise RuntimeError(f"Futu order failed: {data}")

        broker_id = str(data["order_id"].iloc[0])
        order.broker_order_id = broker_id
        order.status = OrderStatus.SUBMITTED
        logger.info("Futu order submitted: %s → %s", order.order_id[:8], broker_id)
        return broker_id

    async def cancel_order(self, broker_order_id: str) -> bool:
        import futu as ft  # type: ignore[import]

        ret, data = self._trade_ctx.modify_order(
            modify_order_op=ft.ModifyOrderOp.CANCEL,
            order_id=int(broker_order_id),
            qty=0,
            price=0,
            trd_env=ft.TrdEnv.REAL if settings.IS_LIVE else ft.TrdEnv.SIMULATE,
            acc_id=int(self._account_id),
        )
        if ret != ft.RET_OK:
            logger.warning("Futu cancel failed: %s", data)
            return False
        return True

    async def get_order_status(self, broker_order_id: str) -> OrderStatus:
        import futu as ft  # type: ignore[import]

        ret, data = self._trade_ctx.order_list_query(
            order_id=int(broker_order_id),
            trd_env=ft.TrdEnv.REAL if settings.IS_LIVE else ft.TrdEnv.SIMULATE,
            acc_id=int(self._account_id),
        )
        if ret != ft.RET_OK or data.empty:
            return OrderStatus.EXPIRED

        status_str = str(data.iloc[0]["order_status"])
        status_map = {
            "SUBMITTED": OrderStatus.SUBMITTED,
            "FILLED_ALL": OrderStatus.FILLED,
            "FILLED_PART": OrderStatus.PARTIAL,
            "CANCELLED_ALL": OrderStatus.CANCELLED,
            "FAILED": OrderStatus.REJECTED,
        }
        return status_map.get(status_str, OrderStatus.SUBMITTED)

    async def get_cash(self) -> float:
        import futu as ft  # type: ignore[import]

        ret, data = self._trade_ctx.accinfo_query(
            trd_env=ft.TrdEnv.REAL if settings.IS_LIVE else ft.TrdEnv.SIMULATE,
            acc_id=int(self._account_id),
        )
        if ret != ft.RET_OK:
            return 0.0
        return float(data.iloc[0].get("cash", 0))

    async def get_positions(self) -> dict:
        import futu as ft  # type: ignore[import]

        ret, data = self._trade_ctx.position_list_query(
            trd_env=ft.TrdEnv.REAL if settings.IS_LIVE else ft.TrdEnv.SIMULATE,
            acc_id=int(self._account_id),
        )
        if ret != ft.RET_OK:
            return {}
        result = {}
        for _, row in data.iterrows():
            code = self._from_futu_code(str(row["code"]))
            result[code] = int(row["qty"])
        return result

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _to_futu_code(symbol: str) -> str:
        """Convert "hk00700" → "HK.00700", "sh600036" → "SH.600036" """
        if symbol.startswith("hk"):
            return f"HK.{symbol[2:]}"
        elif symbol.startswith("sh"):
            return f"SH.{symbol[2:]}"
        elif symbol.startswith("sz"):
            return f"SZ.{symbol[2:]}"
        elif symbol.startswith("us"):
            return f"US.{symbol[2:].upper()}"
        return symbol.upper()

    @staticmethod
    def _from_futu_code(futu_code: str) -> str:
        """Convert "HK.00700" → "hk00700" """
        parts = futu_code.split(".", 1)
        if len(parts) == 2:
            market, code = parts
            return f"{market.lower()}{code}"
        return futu_code.lower()
