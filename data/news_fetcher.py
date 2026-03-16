import finnhub
from config import settings

_client = finnhub.Client(api_key=settings.FINNHUB_API_KEY)


def fetch_headlines(symbol: str, n: int = 20) -> list[str]:
    """Fetches the latest n news headlines for a symbol via Finnhub."""
    from datetime import date, timedelta
    today = date.today().isoformat()
    week_ago = (date.today() - timedelta(days=7)).isoformat()

    news = _client.company_news(symbol, _from=week_ago, to=today)
    headlines = [item["headline"] for item in news if item.get("headline")]
    return headlines[:n]


def fetch_latest_news_item(symbol: str) -> dict | None:
    """
    Returns the single most recent news item for a symbol as a full dict:
    { headline, source, timestamp (ISO-8601), summary, url }
    Returns None if no news found.
    """
    from datetime import date, timedelta, datetime, timezone
    today = date.today().isoformat()
    week_ago = (date.today() - timedelta(days=7)).isoformat()

    news = _client.company_news(symbol, _from=week_ago, to=today)
    if not news:
        return None

    # Finnhub returns items sorted newest-first
    item = news[0]
    ts_unix = item.get("datetime", 0)
    try:
        iso_ts = datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        iso_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "headline":  item.get("headline", ""),
        "source":    item.get("source", "Unknown"),
        "timestamp": iso_ts,
        "summary":   item.get("summary", ""),
        "url":       item.get("url", ""),
    }
