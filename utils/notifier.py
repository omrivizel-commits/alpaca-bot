"""
Notifier — Gmail Email Alerts.

Sends email notifications for:
  - Trade executed (BUY / SELL)
  - HardKillSwitch fired (stop loss)
  - Auto-trade scan summary

Setup: add EMAIL_SENDER, EMAIL_APP_PASSWORD, EMAIL_RECIPIENT to .env
Gmail App Password: myaccount.google.com → Security → 2-Step → App Passwords
"""

import smtplib
import threading
import logging
import asyncio
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger("notifier")

# In-memory log of recent notifications (shown on dashboard)
_notification_log: list[dict] = []
MAX_LOG = 20


def _store(subject: str, body: str):
    _notification_log.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subject": subject,
        "body": body,
    })
    if len(_notification_log) > MAX_LOG:
        _notification_log.pop(0)


def get_log() -> list[dict]:
    return list(reversed(_notification_log))


def is_configured() -> bool:
    from config import settings
    return bool(
        getattr(settings, "EMAIL_SENDER", None) and
        getattr(settings, "EMAIL_APP_PASSWORD", None) and
        getattr(settings, "EMAIL_RECIPIENT", None)
    )


# ── Core send (synchronous) ────────────────────────────────────────────────────

def _send_sync(subject: str, body_html: str):
    from config import settings
    if not is_configured():
        logger.debug("Email not configured — skipping notification.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Omni-Agent] {subject}"
    msg["From"]    = settings.EMAIL_SENDER
    msg["To"]      = settings.EMAIL_RECIPIENT

    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            server.login(settings.EMAIL_SENDER, settings.EMAIL_APP_PASSWORD)
            server.send_message(msg)
        _store(subject, body_html)
        logger.info(f"Email sent: {subject}")
    except Exception as e:
        logger.error(f"Email failed: {e}")


# ── Public API ─────────────────────────────────────────────────────────────────

def notify_sync(subject: str, body_html: str):
    """Fire-and-forget from synchronous code (supervisor, etc.)."""
    threading.Thread(target=_send_sync, args=(subject, body_html), daemon=True).start()


async def notify(subject: str, body_html: str):
    """Async version — use from async code (risk_manager, auto_trader)."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _send_sync, subject, body_html)


# ── Pre-built message templates ───────────────────────────────────────────────

def trade_email(symbol: str, side: str, qty: int, price: float, confidence: float) -> tuple[str, str]:
    color  = "#2ea043" if side == "BUY" else "#da3633"
    emoji  = "🟢" if side == "BUY" else "🔴"
    subject = f"{emoji} {side} {symbol} — {qty} shares @ ${price:.2f}"
    body = f"""
    <div style="font-family:monospace;background:#0d1117;color:#c9d1d9;padding:24px;border-radius:8px">
      <h2 style="color:{color}">{emoji} {side} ORDER EXECUTED</h2>
      <table style="border-collapse:collapse;margin-top:12px">
        <tr><td style="padding:6px 16px 6px 0;color:#8b949e">Symbol</td>    <td><strong>{symbol}</strong></td></tr>
        <tr><td style="padding:6px 16px 6px 0;color:#8b949e">Side</td>      <td style="color:{color}"><strong>{side}</strong></td></tr>
        <tr><td style="padding:6px 16px 6px 0;color:#8b949e">Quantity</td>  <td>{qty} shares</td></tr>
        <tr><td style="padding:6px 16px 6px 0;color:#8b949e">Price</td>     <td>${price:.2f}</td></tr>
        <tr><td style="padding:6px 16px 6px 0;color:#8b949e">Confidence</td><td>{confidence:.1%}</td></tr>
        <tr><td style="padding:6px 16px 6px 0;color:#8b949e">Time</td>      <td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
      </table>
      <p style="margin-top:16px;color:#484f58;font-size:11px">Omni-Agent · Triad of Certainty · Paper Trading</p>
    </div>
    """
    return subject, body


def stop_loss_email(symbol: str, pnl_pct: float, pnl_dollars: float) -> tuple[str, str]:
    subject = f"🛑 STOP LOSS: {symbol} closed at {pnl_pct:.2%}"
    body = f"""
    <div style="font-family:monospace;background:#0d1117;color:#c9d1d9;padding:24px;border-radius:8px">
      <h2 style="color:#da3633">🛑 HARDKILLSWITCH TRIGGERED</h2>
      <table style="border-collapse:collapse;margin-top:12px">
        <tr><td style="padding:6px 16px 6px 0;color:#8b949e">Symbol</td>    <td><strong>{symbol}</strong></td></tr>
        <tr><td style="padding:6px 16px 6px 0;color:#8b949e">P&amp;L %</td> <td style="color:#da3633"><strong>{pnl_pct:.2%}</strong></td></tr>
        <tr><td style="padding:6px 16px 6px 0;color:#8b949e">P&amp;L $</td> <td style="color:#da3633">${pnl_dollars:.2f}</td></tr>
        <tr><td style="padding:6px 16px 6px 0;color:#8b949e">Time</td>      <td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
      </table>
      <p style="margin-top:12px;color:#8b949e">Position force-closed to protect capital.</p>
      <p style="margin-top:16px;color:#484f58;font-size:11px">Omni-Agent · Triad of Certainty · Paper Trading</p>
    </div>
    """
    return subject, body


def scan_summary_email(results: list[dict]) -> tuple[str, str]:
    executed  = [r for r in results if "EXECUTED" in r.get("action", "")]
    buys      = [r for r in results if "BUY"  in r.get("action", "")]
    sells     = [r for r in results if "SELL" in r.get("action", "")]

    trade_count = len(executed)
    subject = (
        f"📈 {trade_count} trade{'s' if trade_count != 1 else ''} executed"
        if trade_count else "🔍 Market scan — no trades today"
    )

    now_str = datetime.now().strftime("%b %d, %Y  %H:%M ET")

    def _conf(r: dict) -> int:
        raw = r.get("master_confidence") or r.get("confidence", 0)
        return int(raw * 100) if isinstance(raw, float) and raw <= 1.0 else int(raw)

    def _conf_color(pct: int) -> str:
        if pct >= 75: return "#16a34a"
        if pct >= 55: return "#ca8a04"
        return "#dc2626"

    def _signal_card(r: dict, side: str) -> str:
        pct   = _conf(r)
        color = "#16a34a" if side == "BUY" else "#dc2626"
        badge = "🟢 BUY"  if side == "BUY" else "🔴 SELL"
        executed_tag = (
            " <span style='background:#16a34a;color:#fff;padding:1px 7px;"
            "border-radius:10px;font-size:11px;margin-left:6px'>EXECUTED</span>"
            if "EXECUTED" in r.get("action", "") else ""
        )
        headline = r.get("top_headline", "")
        sent_dir = r.get("sentiment_direction", "NEUTRAL")
        sent_icon = {"BULLISH": "📰 Bullish", "BEARISH": "📰 Bearish"}.get(sent_dir, "📰 Neutral")
        price_str = f"${r['price']:.2f}" if r.get("price") else "—"
        return f"""
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-left:4px solid {color};
                    border-radius:8px;padding:14px 16px;margin-bottom:10px">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
            <span style="font-size:17px;font-weight:700;color:#111">{r['symbol']}</span>
            <span style="color:{color};font-weight:600">{badge}</span>{executed_tag}
            <span style="margin-left:auto;font-size:13px;color:#6b7280">{price_str}</span>
          </div>
          <div style="margin-bottom:6px">
            <span style="font-size:13px;color:#374151">Confidence: </span>
            <span style="font-size:15px;font-weight:700;color:{_conf_color(pct)}">{pct}%</span>
          </div>
          {f'<div style="font-size:12px;color:#6b7280;margin-bottom:4px">{sent_icon} · {headline[:100]}{"…" if len(headline) > 100 else ""}</div>' if headline else ""}
          <div style="font-size:12px;color:#9ca3af">{r.get("reason","")[:120]}{"…" if len(r.get("reason","")) > 120 else ""}</div>
        </div>"""

    buy_cards  = "".join(_signal_card(r, "BUY")  for r in sorted(buys,  key=_conf, reverse=True))
    sell_cards = "".join(_signal_card(r, "SELL") for r in sorted(sells, key=_conf, reverse=True))

    buy_section = f"""
      <h3 style="color:#16a34a;margin:20px 0 10px;font-size:15px">📈 Buy Signals</h3>
      {buy_cards}
    """ if buys else ""

    sell_section = f"""
      <h3 style="color:#dc2626;margin:20px 0 10px;font-size:15px">📉 Sell Signals</h3>
      {sell_cards}
    """ if sells else ""

    no_signals = """
      <div style="background:#f3f4f6;border-radius:8px;padding:14px 16px;color:#6b7280;font-size:14px">
        No tradeable signals this scan — all symbols filtered by the safety gates.
      </div>
    """ if not buys and not sells else ""

    body = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                max-width:560px;margin:0 auto;background:#ffffff;color:#111827;
                border-radius:12px;overflow:hidden;border:1px solid #e5e7eb">

      <!-- Header -->
      <div style="background:#1e3a5f;padding:20px 24px">
        <div style="font-size:20px;font-weight:700;color:#fff">Omni-Agent · Market Scan</div>
        <div style="font-size:13px;color:#93c5fd;margin-top:4px">{now_str}</div>
      </div>

      <!-- Stats bar -->
      <div style="display:flex;gap:0;border-bottom:1px solid #e5e7eb">
        <div style="flex:1;padding:12px 16px;text-align:center;border-right:1px solid #e5e7eb">
          <div style="font-size:22px;font-weight:700;color:#111">{len(results)}</div>
          <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.5px">Scanned</div>
        </div>
        <div style="flex:1;padding:12px 16px;text-align:center;border-right:1px solid #e5e7eb">
          <div style="font-size:22px;font-weight:700;color:#16a34a">{len(buys)}</div>
          <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.5px">Buy signals</div>
        </div>
        <div style="flex:1;padding:12px 16px;text-align:center;border-right:1px solid #e5e7eb">
          <div style="font-size:22px;font-weight:700;color:#dc2626">{len(sells)}</div>
          <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.5px">Sell signals</div>
        </div>
        <div style="flex:1;padding:12px 16px;text-align:center">
          <div style="font-size:22px;font-weight:700;color:#2563eb">{trade_count}</div>
          <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.5px">Executed</div>
        </div>
      </div>

      <!-- Signal cards -->
      <div style="padding:16px 20px">
        {buy_section}
        {sell_section}
        {no_signals}
      </div>

      <!-- Footer -->
      <div style="padding:12px 20px;background:#f9fafb;border-top:1px solid #e5e7eb;
                  font-size:11px;color:#9ca3af;text-align:center">
        Omni-Agent · Paper Trading · Triad of Certainty
      </div>
    </div>
    """
    return subject, body
