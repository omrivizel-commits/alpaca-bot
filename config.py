from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent / ".env"


class Settings(BaseSettings):
    ALPACA_API_KEY: str
    ALPACA_SECRET_KEY: str
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"

    FINNHUB_API_KEY: str
    GEMINI_API_KEY: str

    RISK_PCT: float = 0.01
    STOP_LOSS_PCT: float = 0.02

    # Email notifications (optional — leave blank to disable)
    EMAIL_SENDER: str | None = None
    EMAIL_APP_PASSWORD: str | None = None
    EMAIL_RECIPIENT: str | None = None

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, **kwargs):
        # .env file takes priority over system environment variables
        return (init_settings, dotenv_settings, env_settings)


settings = Settings()
