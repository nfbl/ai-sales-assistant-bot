import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    bot_token: str
    admin_chat_id: int | None
    llm_provider: str
    llm_api_key: str
    llm_model: str
    llm_base_url: str | None
    llm_reasoning_effort: str | None
    bitrix_webhook: str | None
    nudge_minutes: int
    tz: ZoneInfo
    db_path: Path


def load_settings() -> Settings:
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Не задан BOT_TOKEN в .env — см. README.md")
    admin = os.getenv("ADMIN_CHAT_ID", "").strip()
    webhook = os.getenv("BITRIX_WEBHOOK", "").strip()
    return Settings(
        bot_token=token,
        admin_chat_id=int(admin) if admin else None,
        llm_provider=os.getenv("LLM_PROVIDER", "none").strip().lower(),
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_model=os.getenv("LLM_MODEL", "").strip(),
        llm_base_url=os.getenv("LLM_BASE_URL", "").strip() or None,
        llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "").strip() or None,
        bitrix_webhook=(webhook.rstrip("/") + "/") if webhook else None,
        nudge_minutes=int(os.getenv("NUDGE_MINUTES", "120")),
        tz=ZoneInfo(os.getenv("TIMEZONE", "Europe/Moscow")),
        db_path=BASE_DIR / os.getenv("DB_PATH", "leads.db"),
    )
