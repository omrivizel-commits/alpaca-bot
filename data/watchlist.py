"""
Watchlist Manager — Persistent symbol list.

Stores the active watchlist in watchlist.json next to this file.
Falls back to DEFAULT_SYMBOLS if the file doesn't exist.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("watchlist")

_FILE = Path(__file__).parent.parent / "watchlist.json"

DEFAULT_SYMBOLS = [
    # Validated core — backtested 2y / 3y / 5y out-of-sample
    "AAPL", "NVDA", "TSLA", "PLTR", "AMD",
    "META", "AMZN", "GOOGL", "JPM", "BAC",
    # Lever A expansion (+8) — run 2y backtest before going full-size live
    "NFLX", "COIN", "MU",  "AVGO",
    "GS",   "QQQ",  "V",   "CRM",
]
# MSFT removed : structurally weak across 2y/3y/5y backtests (low ATR, exits before TP)
# PLTR added   : 21-27% returns across 3y/5y out-of-sample validation
# Lever A      : expanded 10 → 18 symbols (2026-03-16) to utilise Gemini free-tier budget
#   NFLX — high ATR, earnings-driven, strong BB-bounce candidate
#   COIN — extreme vol, BB strategy loves wide bands
#   MU   — semiconductor cycle plays, low mega-cap correlation
#   AVGO — semis, distinct momentum profile from AMD/NVDA
#   GS   — finance, different risk profile than JPM/BAC
#   QQQ  — NASDAQ ETF, index-level momentum signal
#   V    — payment rails, low correlation to tech_hv group
#   CRM  — enterprise SaaS, distinct from mega-cap tech

# Extended universe users can add from
UNIVERSE = [
    "AAPL","NVDA","TSLA","PLTR","AMD","META","AMZN","GOOGL","JPM","BAC",
    "NFLX","CRM","ADBE","INTC","QCOM","MU","AVGO","TXN","ORCL","IBM",
    "GS","MS","C","WFC","BRK-B","V","MA","PYPL","SQ","COIN",
    "XOM","CVX","COP","SLB","OXY","UNH","JNJ","PFE","MRNA","ABBV",
    "SPY","QQQ","DIA","IWM","GLD","SLV","TLT","HYG","MSFT",
]


def load() -> list[str]:
    if _FILE.exists():
        try:
            return json.loads(_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_SYMBOLS.copy()


def save(symbols: list[str]):
    _FILE.write_text(json.dumps(symbols, indent=2), encoding="utf-8")
    logger.info(f"Watchlist saved: {symbols}")


def add_symbol(symbol: str) -> list[str]:
    symbol = symbol.upper().strip()
    wl = load()
    if symbol not in wl:
        wl.append(symbol)
        save(wl)
    return wl


def remove_symbol(symbol: str) -> list[str]:
    symbol = symbol.upper().strip()
    wl = load()
    wl = [s for s in wl if s != symbol]
    save(wl)
    return wl
