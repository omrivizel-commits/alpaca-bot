"""
Alpaca Broker — Paper Trading Execution Layer.
Uses the official alpaca-py SDK (replaces deprecated alpaca-trade-api).
"""

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import settings

_client: TradingClient | None = None


def _get_client() -> TradingClient:
    global _client
    if _client is None:
        _client = TradingClient(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY,
            paper=True,
        )
    return _client


def submit_order(symbol: str, qty: int, side: str) -> dict:
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    order = _get_client().submit_order(req)
    return {
        "order_id": str(order.id),
        "symbol": order.symbol,
        "qty": str(order.qty),
        "side": order.side.value,
        "status": order.status.value,
        "submitted_at": str(order.submitted_at),
    }


def close_position(symbol: str) -> dict | None:
    try:
        result = _get_client().close_position(symbol)
        return {
            "order_id": str(result.id),
            "symbol": symbol,
            "status": "closed",
            "submitted_at": str(result.submitted_at),
        }
    except Exception:
        return None


def get_position(symbol: str) -> dict | None:
    try:
        pos = _get_client().get_open_position(symbol)
        return {
            "symbol": pos.symbol,
            "qty": float(pos.qty),
            "avg_entry_price": float(pos.avg_entry_price),
            "current_price": float(pos.current_price),
            "unrealized_pnl": float(pos.unrealized_pl),
            "unrealized_pnl_pct": float(pos.unrealized_plpc),
        }
    except Exception:
        return None


def get_portfolio_value() -> float:
    account = _get_client().get_account()
    return float(account.equity)


def _safe_float(val) -> float | None:
    """Converts a value to float, returning None if null or unparseable."""
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def list_positions() -> list[dict]:
    positions = _get_client().get_all_positions()
    return [
        {
            "symbol": p.symbol,
            "qty": _safe_float(p.qty),
            "avg_entry_price": _safe_float(p.avg_entry_price),
            "current_price": _safe_float(p.current_price),
            "unrealized_pnl": _safe_float(p.unrealized_pl),
            "unrealized_pnl_pct": _safe_float(p.unrealized_plpc),
        }
        for p in positions
    ]
