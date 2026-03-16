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
    executed = [r for r in results if "EXECUTED" in r.get("action", "")]
    subject  = f"📊 Scan complete — {len(executed)} trade(s) executed"
    rows = "".join(
        f"<tr><td style='padding:5px 12px 5px 0'><strong>{r['symbol']}</strong></td>"
        f"<td style='padding:5px 12px 5px 0;color:{'#2ea043' if 'EXECUTED' in r.get('action','') else '#8b949e'}'>{r.get('action','—')}</td>"
        f"<td style='padding:5px 12px 5px 0;color:#8b949e'>{r.get('reason','')[:60]}</td></tr>"
        for r in results
    )
    body = f"""
    <div style="font-family:monospace;background:#0d1117;color:#c9d1d9;padding:24px;border-radius:8px">
      <h2 style="color:#58a6ff">📊 Auto-Trade Scan Complete</h2>
      <p style="color:#8b949e;margin:8px 0">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · {len(results)} symbols scanned · {len(executed)} trades executed</p>
      <table style="border-collapse:collapse;margin-top:12px">{rows}</table>
      <p style="margin-top:16px;color:#484f58;font-size:11px">Omni-Agent · Triad of Certainty · Paper Trading</p>
    </div>
    """
    return subject, body
