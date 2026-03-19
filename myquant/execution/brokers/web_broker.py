"""
WebBroker — Playwright-based broker that automates a web trading platform.

Default selectors target 中山证券 (Zhongshan Securities) H5 web trading.
To adapt to another broker, subclass WebBroker and override _cfg, or
pass a custom WebBrokerConfig instance to the constructor.

Setup (one-time):
    pip install playwright
    playwright install chromium

Required env vars:
    WEB_BROKER_URL        — e.g. https://trade.zszq.com/h5/
    WEB_BROKER_USERNAME   — account number / ID card number
    WEB_BROKER_PASSWORD   — trading password
    WEB_BROKER_HEADLESS   — "true" (default) | "false" for debugging

Usage:
    broker = WebBroker()
    await broker.connect()
    await broker.submit_order(order)
    await broker.disconnect()
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from myquant.config.settings import settings
from myquant.execution.brokers.base_broker import BaseBroker
from myquant.models.order import Order, OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)

# ── Selector configuration ────────────────────────────────────────────────────
# All CSS/text selectors are gathered here so they are easy to update when
# the broker redesigns their web UI without touching any logic code.


@dataclass(frozen=True)
class WebBrokerConfig:
    """
    Holds every CSS/XPath selector and URL used by WebBroker.

    Defaults are for 中山证券 H5 web trading.
    Override individual fields when constructing for another broker.

    How to find the right selectors:
      1. Open the broker's web trading URL in Chrome.
      2. Log in manually and open DevTools (F12).
      3. Use the Inspector (Ctrl+Shift+C) to click each element and
         note its id / class / data-* attribute.
      4. Update the relevant field below.
    """

    # ── Entry URL ─────────────────────────────────────────────
    base_url: str = field(default_factory=lambda: settings.WEB_BROKER_URL)

    # ── Login page ────────────────────────────────────────────
    # The username field (account number / ID card)
    login_username_sel: str = "input[name='loginName'], input[placeholder*='账号'], #loginName"
    # The password field
    login_password_sel: str = "input[type='password'], input[placeholder*='密码'], #loginPassword"
    # The submit / login button
    login_button_sel: str   = "button[type='submit'], .login-btn, #loginBtn"
    # An element that only appears after successful login (used to confirm auth)
    login_success_sel: str  = ".account-name, .user-info, .nav-account, [class*='username']"

    # ── Order entry ───────────────────────────────────────────
    # Navigation: how to reach the "Buy" order-entry page
    buy_nav_sel: str  = "a[href*='buy'], .menu-buy, [data-action='buy']"
    sell_nav_sel: str = "a[href*='sell'], .menu-sell, [data-action='sell']"

    # Input fields on the order form
    order_symbol_sel: str = "input[name='stockCode'], #stockCode, input[placeholder*='股票代码']"
    order_qty_sel: str    = "input[name='tradeNumber'], #tradeNumber, input[placeholder*='数量']"
    order_price_sel: str  = "input[name='tradePrice'], #tradePrice, input[placeholder*='价格']"

    # "市价" / "限价" toggle — click to switch between market and limit
    market_order_toggle_sel: str = ".price-type-market, [data-type='market'], label[for*='market']"
    limit_order_toggle_sel: str  = ".price-type-limit,  [data-type='limit'],  label[for*='limit']"

    # Final submit button ("买入" / "卖出")
    order_submit_sel: str = ".trade-submit, button.submit, #submitBtn, button[type='submit']"

    # Confirmation dialog that appears after submit (click OK/确认 to confirm)
    order_confirm_sel: str = ".confirm-btn, .dialog-confirm, button:has-text('确认'), button:has-text('确定')"

    # Toast / banner that shows the order ID after submission
    order_success_sel: str = ".success-tip, .order-result, [class*='success']"

    # ── Position / account ────────────────────────────────────
    # Navigation to the portfolio / holdings page
    portfolio_nav_sel: str = "a[href*='position'], .menu-position, [data-page='position']"
    # Each row in the positions table
    position_row_sel: str  = ".position-item, tr.holding-row, [class*='position-row']"
    # Within each row: the symbol code and the quantity held
    position_symbol_attr: str = "data-code"      # or use a sub-selector
    position_symbol_sel: str  = ".stock-code, td.code, [class*='stockCode']"
    position_qty_sel: str     = ".hold-volume, td.qty, [class*='holdVol']"

    # Navigation to the account / cash page
    account_nav_sel: str = "a[href*='account'], .menu-account, [data-page='account']"
    # Element that shows available cash balance
    cash_sel: str        = ".available-money, .cash-amount, [class*='availableMoney']"

    # ── Order history ─────────────────────────────────────────
    order_history_nav_sel: str = "a[href*='orders'], .menu-orders, [data-page='orders']"
    order_row_sel: str         = ".order-item, tr.order-row"
    order_id_sel: str          = "[data-order-id], .order-id, td.orderId"
    order_status_sel: str      = ".order-status, td.status, [class*='orderStatus']"


# ── Default config (中山证券) ─────────────────────────────────────────────────

_DEFAULT_CFG = WebBrokerConfig()

# ── Status mapping ────────────────────────────────────────────────────────────

_STATUS_MAP: dict[str, OrderStatus] = {
    # Chinese status strings → internal OrderStatus
    "已成": OrderStatus.FILLED,
    "全部成交": OrderStatus.FILLED,
    "部分成交": OrderStatus.PARTIAL,
    "已撤": OrderStatus.CANCELLED,
    "已撤单": OrderStatus.CANCELLED,
    "废单": OrderStatus.REJECTED,
    "未报": OrderStatus.SUBMITTED,
    "已报": OrderStatus.SUBMITTED,
    "待成交": OrderStatus.SUBMITTED,
    # English fallbacks (some brokers show bilingual UI)
    "filled":    OrderStatus.FILLED,
    "partial":   OrderStatus.PARTIAL,
    "cancelled": OrderStatus.CANCELLED,
    "rejected":  OrderStatus.REJECTED,
}


# ── WebBroker ─────────────────────────────────────────────────────────────────

class WebBroker(BaseBroker):
    """
    Playwright-based broker adapter.

    Maintains a single persistent browser session so that login only
    happens once.  All order and account methods reuse the same page.

    Parameters
    ----------
    cfg :
        Selector configuration.  Defaults to 中山证券 H5.
    headless :
        Run browser without a visible window (recommended in production).
        Set to False while debugging selectors.
    slow_mo :
        Milliseconds between Playwright actions — useful for debugging.
    """

    def __init__(
        self,
        cfg: WebBrokerConfig = _DEFAULT_CFG,
        headless: Optional[bool] = None,
        slow_mo: int = 150,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._headless = headless if headless is not None else settings.WEB_BROKER_HEADLESS
        self._slow_mo = slow_mo
        self._playwright = None
        self._browser = None
        self._page = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Launch browser and log in to the broker website."""
        try:
            from playwright.async_api import async_playwright  # type: ignore[import]
        except ImportError:
            raise RuntimeError(
                "playwright not installed.\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            slow_mo=self._slow_mo,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await self._browser.new_context(
            # Spoof a normal desktop browser UA to avoid bot detection
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="zh-CN",
        )
        self._page = await context.new_page()

        await self._login()
        logger.info("WebBroker connected to %s", self._cfg.base_url)

    async def disconnect(self) -> None:
        """Close the browser session cleanly."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._page = None
        self._browser = None
        self._playwright = None
        logger.info("WebBroker disconnected.")

    # ── Trading ───────────────────────────────────────────────────────────────

    async def submit_order(self, order: Order) -> str:
        """
        Automate order entry on the web trading form.

        Returns the broker-assigned order ID (captured from the success toast).
        """
        self._assert_connected()

        # Navigate to buy or sell page
        if order.side == OrderSide.BUY:
            await self._nav_to(self._cfg.buy_nav_sel, "买入")
        else:
            await self._nav_to(self._cfg.sell_nav_sel, "卖出")

        # --- Symbol ---
        symbol_code = _to_exchange_code(order.symbol)
        await self._fill(self._cfg.order_symbol_sel, symbol_code)
        # Wait for auto-complete / name lookup to settle
        await self._page.wait_for_timeout(600)

        # --- Order type toggle ---
        if order.order_type == OrderType.MARKET:
            await self._click_if_present(self._cfg.market_order_toggle_sel)
        else:
            await self._click_if_present(self._cfg.limit_order_toggle_sel)

        # --- Price (limit orders only) ---
        if order.order_type == OrderType.LIMIT and order.limit_price:
            await self._fill(self._cfg.order_price_sel, str(order.limit_price))

        # --- Quantity ---
        await self._fill(self._cfg.order_qty_sel, str(order.quantity))

        # --- Submit ---
        await self._page.click(self._cfg.order_submit_sel)

        # --- Confirm dialog (some brokers show a "are you sure?" popup) ---
        try:
            await self._page.wait_for_selector(self._cfg.order_confirm_sel, timeout=3000)
            await self._page.click(self._cfg.order_confirm_sel)
        except Exception:
            pass  # No confirmation dialog — that's fine

        # --- Capture broker order ID from success notification ---
        broker_id = await self._capture_order_id()
        order.broker_order_id = broker_id
        order.status = OrderStatus.SUBMITTED
        logger.info(
            "WebBroker submitted %s %s ×%d → broker_id=%s",
            order.side.value, order.symbol, order.quantity, broker_id,
        )
        return broker_id

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Navigate to order history and cancel an open order."""
        self._assert_connected()
        try:
            await self._nav_to(self._cfg.order_history_nav_sel, "委托")
            page = self._page

            # Find the row whose order-id attribute matches
            rows = await page.query_selector_all(self._cfg.order_row_sel)
            for row in rows:
                oid_el = await row.query_selector(self._cfg.order_id_sel)
                if oid_el:
                    oid_text = (await oid_el.inner_text()).strip()
                    if oid_text == broker_order_id:
                        cancel_btn = await row.query_selector(
                            "button:has-text('撤单'), .cancel-btn, [data-action='cancel']"
                        )
                        if cancel_btn:
                            await cancel_btn.click()
                            await self._page.wait_for_timeout(800)
                            logger.info("WebBroker cancel sent for %s", broker_order_id)
                            return True
        except Exception as exc:
            logger.warning("WebBroker cancel failed: %s", exc)
        return False

    async def get_order_status(self, broker_order_id: str) -> OrderStatus:
        """Scrape the order list to find the current status of an order."""
        self._assert_connected()
        try:
            await self._nav_to(self._cfg.order_history_nav_sel, "委托")
            rows = await self._page.query_selector_all(self._cfg.order_row_sel)
            for row in rows:
                oid_el = await row.query_selector(self._cfg.order_id_sel)
                if oid_el and (await oid_el.inner_text()).strip() == broker_order_id:
                    status_el = await row.query_selector(self._cfg.order_status_sel)
                    if status_el:
                        raw = (await status_el.inner_text()).strip()
                        return _STATUS_MAP.get(raw, OrderStatus.SUBMITTED)
        except Exception as exc:
            logger.warning("get_order_status error: %s", exc)
        return OrderStatus.EXPIRED

    async def get_cash(self) -> float:
        """Scrape available cash from the account page."""
        self._assert_connected()
        try:
            await self._nav_to(self._cfg.account_nav_sel, "账户")
            el = await self._page.wait_for_selector(self._cfg.cash_sel, timeout=5000)
            text = (await el.inner_text()).strip().replace(",", "").replace("￥", "").replace("¥", "")
            return float(text)
        except Exception as exc:
            logger.warning("get_cash error: %s", exc)
            return 0.0

    async def get_positions(self) -> dict[str, int]:
        """Scrape current holdings from the portfolio page."""
        self._assert_connected()
        positions: dict[str, int] = {}
        try:
            await self._nav_to(self._cfg.portfolio_nav_sel, "持仓")
            rows = await self._page.query_selector_all(self._cfg.position_row_sel)
            for row in rows:
                sym_el = await row.query_selector(self._cfg.position_symbol_sel)
                qty_el = await row.query_selector(self._cfg.position_qty_sel)
                if sym_el and qty_el:
                    raw_code = (await sym_el.inner_text()).strip()
                    raw_qty  = (await qty_el.inner_text()).strip().replace(",", "")
                    symbol   = _from_exchange_code(raw_code)
                    try:
                        positions[symbol] = int(float(raw_qty))
                    except ValueError:
                        pass
        except Exception as exc:
            logger.warning("get_positions error: %s", exc)
        return positions

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _login(self) -> None:
        page = self._page
        await page.goto(self._cfg.base_url, wait_until="networkidle", timeout=30_000)

        # Fill credentials
        await self._fill(self._cfg.login_username_sel, settings.WEB_BROKER_USERNAME)
        await self._fill(self._cfg.login_password_sel, settings.WEB_BROKER_PASSWORD)
        await page.click(self._cfg.login_button_sel)

        # Wait until a post-login element is visible
        try:
            await page.wait_for_selector(self._cfg.login_success_sel, timeout=15_000)
            logger.info("WebBroker login successful.")
        except Exception:
            # Take a screenshot to aid debugging if login fails
            await page.screenshot(path="web_broker_login_fail.png")
            raise RuntimeError(
                "WebBroker login failed — page did not reach post-login state.\n"
                "Screenshot saved to web_broker_login_fail.png.\n"
                "Set WEB_BROKER_HEADLESS=false to debug visually."
            )

    async def _nav_to(self, selector: str, label: str = "") -> None:
        """Click a navigation link; skip if already on that page."""
        page = self._page
        try:
            nav_el = await page.query_selector(selector)
            if nav_el:
                await nav_el.click()
                await page.wait_for_load_state("networkidle", timeout=10_000)
            else:
                logger.debug("Nav element '%s' not found (label=%s) — already on page?", selector, label)
        except Exception as exc:
            logger.debug("_nav_to('%s') skipped: %s", label, exc)

    async def _fill(self, selector: str, value: str) -> None:
        """Clear an input field and type a value."""
        el = await self._page.wait_for_selector(selector, timeout=5000)
        await el.triple_click()   # select all existing text
        await el.type(value, delay=60)

    async def _click_if_present(self, selector: str) -> None:
        el = await self._page.query_selector(selector)
        if el:
            await el.click()

    async def _capture_order_id(self) -> str:
        """Try to read the order ID from the success notification; fall back to a local UUID."""
        try:
            el = await self._page.wait_for_selector(self._cfg.order_success_sel, timeout=5000)
            text = (await el.inner_text()).strip()
            # Many brokers include the order number in the toast, e.g. "委托成功 编号: 20240318001"
            import re
            m = re.search(r"[\d]{6,}", text)
            if m:
                return m.group()
        except Exception:
            pass
        return f"WEB-{str(uuid.uuid4())[:8].upper()}"

    def _assert_connected(self) -> None:
        if self._page is None:
            raise RuntimeError("WebBroker not connected. Call await broker.connect() first.")


# ── Symbol code helpers ───────────────────────────────────────────────────────

def _to_exchange_code(symbol: str) -> str:
    """
    Convert internal symbol format to what the broker's search box expects.

    Examples:
        "sh600519"  →  "600519"   (A-share, SH prefix implied by code range)
        "sz300750"  →  "300750"
        "hk00700"   →  "00700"    (HK — broker may need the full 5-digit code)
    """
    for prefix in ("sh", "sz", "hk", "us"):
        if symbol.startswith(prefix):
            return symbol[len(prefix):]
    return symbol


def _from_exchange_code(code: str) -> str:
    """
    Guess the internal prefix from a bare numeric code scraped from the UI.

    Examples:
        "600519" → "sh600519"   (SH range: 600xxx, 601xxx, 603xxx, 688xxx)
        "300750" → "sz300750"   (SZ range: 000xxx, 002xxx, 300xxx)
        "00700"  → "hk00700"
    """
    code = code.strip()
    if code.startswith(("6", "0")):
        # Distinguish SH 60xxxx from SZ 00xxxx/002xxx
        if code.startswith("6"):
            return f"sh{code}"
        return f"sz{code}"
    if code.startswith("3"):
        return f"sz{code}"
    if len(code) == 5 and code.isdigit():
        return f"hk{code}"
    return code.lower()
