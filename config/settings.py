import os

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )

    return value


DISCORD_TOKEN: str = _required("DISCORD_TOKEN")
GEMINI_API_KEY: str = _required("GEMINI_API_KEY")
COHERE_API_KEY: str = _required("COHERE_API_KEY")


def validate_settings() -> None:
    # Importing this module already validates the required settings.
    return None