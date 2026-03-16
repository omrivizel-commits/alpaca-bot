"""
Risk Manager — The "Armor."

Two responsibilities:
1. Position Sizing: Never bet more than RISK_PCT (1%) of portfolio per trade.
2. HardKillSwitch: Background monitor that force-closes any position that
   drops more than STOP_LOSS_PCT (2%), bypassing the Supervisor entirely.
"""

import asyncio
import logging
from config import settings
from execution.alpaca_broker import (
    get_portfolio_value,
    get_position,
    close_position,
    list_positions,
)

logger = logging.getLogger("risk_manager")


def calculate_position_size(price: float) -> int:
    """
    Returns the number of whole shares to buy, risking at most RISK_PCT of portfolio.

    Formula: floor((portfolio_value * RISK_PCT) / price)
    """
    portfolio_value = get_portfolio_value()
    max_dollars = portfolio_value * settings.RISK_PCT
    qty = int(max_dollars / price)
    return max(qty, 1)  # Always at least 1 share if affordable


class HardKillSwitch:
    """
    Async background task that polls every 30 seconds and force-closes
    any position that has lost more than STOP_LOSS_PCT.
    """

    def __init__(self, poll_interval: int = 30):
        self.poll_interval = poll_interval
        self._running = False

    async def start(self):
        self._running = True
        logger.info("HardKillSwitch armed and monitoring.")
        while self._running:
            await self._check_all_positions()
            await asyncio.sleep(self.poll_interval)

    def stop(self):
        self._running = False
        logger.info("HardKillSwitch disarmed.")

    async def _check_all_positions(self):
        loop = asyncio.get_running_loop()
        positions = await loop.run_in_executor(None, list_positions)

        for pos in positions:
            pnl_pct = pos["unrealized_pnl_pct"]  # e.g. -0.025 = -2.5%
            if pnl_pct <= -settings.STOP_LOSS_PCT:
                symbol = pos["symbol"]
                pnl_dollars = pos.get("unrealized_pnl") or 0.0
                logger.warning(
                    f"KILL SWITCH TRIGGERED: {symbol} down {pnl_pct:.2%}. "
                    f"Force-closing position."
                )
                result = await loop.run_in_executor(None, close_position, symbol)
                logger.warning(f"Position closed: {result}")
                # Record LOSS post-mortem and trigger self-review if ready
                try:
                    from utils.post_mortem import update_trade_result, auto_review_if_ready
                    exit_price = float(pos.get("current_price", 0) or pos.get("avg_entry_price", 0))
                    if exit_price:
                        update_trade_result(symbol, exit_price, "LOSS")
                    review = auto_review_if_ready(10)
                    if review:
                        logger.info(f"Auto-review triggered after kill switch:\n{review}")
                except Exception as _pm_err:
                    logger.warning(f"Post-mortem error [{symbol}]: {_pm_err}")
                # Send stop-loss email
                try:
                    from utils.notifier import notify, stop_loss_email
                    subj, body = stop_loss_email(symbol, pnl_pct, pnl_dollars)
                    await notify(subj, body)
                except Exception:
                    pass
