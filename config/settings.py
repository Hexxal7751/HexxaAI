import os

from dotenv import load_dotenv

load_dotenv()


DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def validate_settings() -> None:
    missing = []

    if not DISCORD_TOKEN:
        missing.append("DISCORD_TOKEN")

    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )