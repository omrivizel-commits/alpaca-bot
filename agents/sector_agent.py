"""
Sector Agent — Pure Mathematical Sector Rotation Analyzer.

Fetches live sector ETF data + constituent stock snapshots, then applies
a deterministic momentum-scoring formula to identify:
  - Hottest / coldest sectors by intraday momentum
  - Top 3-5 stocks per hot sector with day-trade scores (0-100)
  - Entry opportunities with specific timeframes
  - Sector catalyst and caution flags

Data fetched via yfinance. No AI calls required.

Day-trade score formula (0-100):
  momentum_component  = min(40, |change_pct| × 8)
  volume_component    = min(30, (vol_ratio − 1) × 15)  when vol_ratio > 1
  rsi_component       = max(0,  20 × (1 − |rsi − 55| / 45))   peak at RSI ≈ 55
  trend_component     = 10 if above MA-50 else 0
  score               = momentum + volume + rsi + trend   [clamped 0-100]
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Sector universe ───────────────────────────────────────────────────────────

SECTOR_ETFS: dict = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLE":  "Energy",
    "XLI":  "Industrials",
    "XLY":  "Consumer Disc",
    "XLC":  "Communication",
    "XLU":  "Utilities",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
}

SECTOR_CONSTITUENTS: dict = {
    "XLK":  ["AAPL", "MSFT", "NVDA", "META", "AVGO", "ORCL", "ADBE", "AMD"],
    "XLF":  ["JPM",  "BAC",  "WFC",  "GS",   "MS",   "C",    "AXP",  "BLK"],
    "XLV":  ["LLY",  "UNH",  "JNJ",  "ABBV", "MRK",  "PFE",  "TMO",  "ABT"],
    "XLE":  ["XOM",  "CVX",  "COP",  "EOG",  "SLB",  "PSX",  "MPC",  "OXY"],
    "XLI":  ["CAT",  "HON",  "UPS",  "RTX",  "BA",   "LMT",  "DE",   "GE"],
    "XLY":  ["AMZN", "TSLA", "HD",   "MCD",  "NKE",  "LOW",  "SBUX", "BKNG"],
    "XLC":  ["GOOGL","NFLX", "DIS",  "CMCSA","T",    "VZ",   "TMUS", "CHTR"],
    "XLU":  ["NEE",  "DUK",  "SO",   "D",    "AEP",  "EXC",  "SRE",  "XEL"],
    "XLB":  ["LIN",  "APD",  "ECL",  "SHW",  "NEM",  "FCX",  "NUE",  "VMC"],
    "XLRE": ["PLD",  "AMT",  "CCI",  "EQIX", "PSA",  "SPG",  "O",    "DLR"],
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    val   = rsi.iloc[-1]
    return round(float(val), 1) if not np.isnan(val) else 50.0


def _fetch_snapshot(symbol: str, need_rsi: bool = False) -> dict:
    """Fetches a 60-day daily snapshot for a symbol."""
    try:
        df = yf.Ticker(symbol).history(period="60d", interval="1d", auto_adjust=True)
        if df is None or len(df) < 3:
            return None

        close      = df["Close"]
        price      = round(float(close.iloc[-1]), 2)
        prev       = float(close.iloc[-2])
        change_pct = round((price - prev) / prev * 100, 2) if prev else 0.0
        volume     = int(df["Volume"].iloc[-1])
        vol_ma     = int(df["Volume"].rolling(20).mean().iloc[-1])

        snap = {
            "price":      price,
            "change_pct": change_pct,
            "volume":     volume,
            "volume_ma":  vol_ma,
        }

        if need_rsi:
            ma50_val = close.rolling(50).mean().iloc[-1]
            snap["rsi"]        = _compute_rsi(close)
            snap["above_ma50"] = bool(price > ma50_val) \
                                 if not np.isnan(ma50_val) else False

        return snap
    except Exception:
        return None


def _fetch_all_sectors() -> dict:
    """Fetches snapshots for all sector ETFs in parallel."""
    results: dict = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_snapshot, etf): etf for etf in SECTOR_ETFS}
        for future in as_completed(futures):
            etf  = futures[future]
            snap = future.result()
            if snap:
                results[etf] = snap
    return results


def _fetch_stocks_for_sectors(sector_etfs: list, max_per_sector: int = 5) -> dict:
    """Fetches stock-level snapshots for the given sectors in parallel."""
    all_symbols = []
    for etf in sector_etfs:
        for ticker in SECTOR_CONSTITUENTS.get(etf, [])[:max_per_sector]:
            all_symbols.append((etf, ticker))

    per_etf = {etf: [] for etf in sector_etfs}

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_fetch_snapshot, ticker, True): (etf, ticker)
                   for etf, ticker in all_symbols}
        for future in as_completed(futures):
            etf, ticker = futures[future]
            snap = future.result()
            if snap:
                per_etf[etf].append({"ticker": ticker, **snap})

    for etf in per_etf:
        per_etf[etf].sort(key=lambda x: abs(x.get("change_pct", 0)), reverse=True)

    return per_etf


# ── Pure-math interpretation ──────────────────────────────────────────────────

def _day_trade_score(snap: dict) -> int:
    """
    Computes 0-100 day-trade score for a single stock snapshot.
    """
    abs_change = abs(snap.get("change_pct", 0.0))
    vol_ma     = snap.get("volume_ma",  1)
    volume     = snap.get("volume",     0)
    rsi        = snap.get("rsi",        50.0)
    above_ma50 = snap.get("above_ma50", False)

    vol_ratio = volume / vol_ma if vol_ma > 0 else 1.0

    momentum_component = min(40, abs_change * 8)
    volume_component   = min(30, (vol_ratio - 1) * 15) if vol_ratio > 1 else 0
    rsi_component      = max(0.0, 20 * (1 - abs(rsi - 55) / 45))
    trend_component    = 10 if above_ma50 else 0

    return int(min(100, momentum_component + volume_component
                        + rsi_component + trend_component))


def _interpret_sector_rotation(
    sector_snaps: dict,
    ranked_etfs: list,
    hot_etfs: list,
    stock_data: dict,
) -> dict:
    """
    Applies pure scoring logic to produce a full sector rotation analysis.
    """
    # ── Sector ranking ────────────────────────────────────────────────────────
    sector_ranking = []
    for i, etf in enumerate(ranked_etfs):
        snap   = sector_snaps.get(etf, {})
        change = snap.get("change_pct", 0.0)
        if   change > 1.0:  momentum = "strong"
        elif change > 0.3:  momentum = "moderate"
        elif change > 0.0:  momentum = "weak"
        else:               momentum = "negative"
        sector_ranking.append({
            "etf":        etf,
            "change_pct": change,
            "momentum":   momentum,
            "rank":       i + 1,
        })

    # ── Score and rank stocks ─────────────────────────────────────────────────
    all_stocks = []
    for etf in hot_etfs:
        for stock in stock_data.get(etf, []):
            vol_ma    = stock.get("volume_ma",  1)
            volume    = stock.get("volume",     0)
            vol_ratio = volume / vol_ma if vol_ma > 0 else 1.0
            dt_score  = _day_trade_score(stock)
            change_pct = stock.get("change_pct", 0.0)

            if   dt_score >= 80: setup = "excellent"
            elif dt_score >= 65: setup = "very_good"
            elif dt_score >= 50: setup = "good"
            elif dt_score >= 35: setup = "fair"
            else:                setup = "poor"

            technical = (f"{'+' if change_pct >= 0 else ''}{change_pct:.1f}% "
                         f"on {vol_ratio:.1f}×volume")

            if change_pct > 0:
                entry_opp = (f"Long above {stock['price'] * 1.002:.2f} "
                             "on momentum continuation")
                t1 = round(stock["price"] * 1.010, 2)
                t2 = round(stock["price"] * 1.020, 2)
            else:
                entry_opp = (f"Short below {stock['price'] * 0.998:.2f} "
                             "on momentum breakdown")
                t1 = round(stock["price"] * 0.990, 2)
                t2 = round(stock["price"] * 0.980, 2)

            risk_level = ("high"     if abs(change_pct) > 3 else
                          "moderate" if abs(change_pct) > 1 else "low")

            all_stocks.append({
                "ticker":            stock["ticker"],
                "price":             stock["price"],
                "change_pct":        change_pct,
                "day_trade_score":   dt_score,
                "setup_quality":     setup,
                "technical_status":  technical,
                "entry_opportunity": entry_opp,
                "target_1":          t1,
                "target_2":          t2,
                "risk_level":        risk_level,
            })

    all_stocks.sort(key=lambda x: x["day_trade_score"], reverse=True)
    top_stocks = all_stocks[:10]

    # ── Sector catalyst text ──────────────────────────────────────────────────
    hottest    = hot_etfs[0] if hot_etfs else "XLK"
    hot_snap   = sector_snaps.get(hottest, {})
    hot_change = hot_snap.get("change_pct", 0.0)

    if   hot_change > 2.0:
        catalyst = (f"Strong momentum in {SECTOR_ETFS.get(hottest, hottest)} "
                    f"sector (+{hot_change:.1f}%)")
    elif hot_change > 0.5:
        catalyst = (f"Moderate rotation into {SECTOR_ETFS.get(hottest, hottest)} "
                    f"(+{hot_change:.1f}%)")
    elif hot_change > 0.0:
        catalyst = (f"Mild strength in {SECTOR_ETFS.get(hottest, hottest)} "
                    f"(+{hot_change:.1f}%)")
    elif hot_change < -1.0:
        catalyst = (f"Sector weakness: {SECTOR_ETFS.get(hottest, hottest)} "
                    f"({hot_change:.1f}%)")
    else:
        catalyst = "Flat sector rotation — mixed signals"

    # ── Trading window ────────────────────────────────────────────────────────
    if   abs(hot_change) > 2.0: window = "Active for 1-2 hours post open"
    elif abs(hot_change) > 1.0: window = "30-60 minutes of opportunity"
    else:                       window = "Limited — low sector momentum"

    # ── Caution flags ─────────────────────────────────────────────────────────
    negative_count = sum(
        1 for etf, snap in sector_snaps.items()
        if snap.get("change_pct", 0) < -1.0
    )
    if negative_count > 5:
        caution = "Most sectors negative — broad market weakness, use caution."
    elif abs(hot_change) < 0.3:
        caution = "Low sector momentum — scalp conditions only, avoid large positions."
    else:
        caution = ""

    return {
        "hottest_sector":         hottest,
        "sector_ranking":         sector_ranking,
        "top_stocks_in_rotation": top_stocks,
        "sector_catalyst":        catalyst,
        "trading_window":         window,
        "caution":                caution,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_sector_rotation(top_n_sectors: int = 3) -> dict:
    """
    Fetches sector ETF data, finds the top sectors by momentum, pulls
    constituent stock snapshots, and identifies the best day-trade setups
    using deterministic scoring — no AI calls.

    Returns a full rotation analysis with ranked sectors and stock picks.
    """
    _fallback = {
        "hottest_sector":         "UNKNOWN",
        "sector_ranking":         [],
        "top_stocks_in_rotation": [],
        "sector_catalyst":        "Data unavailable",
        "trading_window":         "Unknown",
        "caution":                "Sector scan failed — check market data connection.",
    }

    # ── Step 1: Fetch all sector ETF data ─────────────────────────────────────
    sector_snaps = _fetch_all_sectors()
    if not sector_snaps:
        return _fallback

    # ── Step 2: Rank by momentum score (change_pct × volume ratio) ───────────
    def _momentum_score(etf: str) -> float:
        s         = sector_snaps[etf]
        vol_ratio = s["volume"] / s["volume_ma"] if s.get("volume_ma") else 1.0
        return s["change_pct"] * vol_ratio

    ranked_etfs = sorted(sector_snaps.keys(), key=_momentum_score, reverse=True)
    hot_etfs    = ranked_etfs[:top_n_sectors]

    # ── Step 3: Fetch stock data for top sectors ──────────────────────────────
    stock_data = _fetch_stocks_for_sectors(hot_etfs, max_per_sector=5)

    # ── Step 4: Pure-math interpretation ──────────────────────────────────────
    result = _interpret_sector_rotation(sector_snaps, ranked_etfs, hot_etfs, stock_data)

    # Attach raw sector data for dashboard display
    result["_sector_data"] = {
        etf: {
            "change_pct": sector_snaps[etf]["change_pct"],
            "name":       SECTOR_ETFS.get(etf, etf),
        }
        for etf in sector_snaps
    }
    result["hottest_sector"] = hot_etfs[0] if hot_etfs else "UNKNOWN"

    return result
