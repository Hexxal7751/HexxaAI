from dataclasses import dataclass, field
from enum import Enum


class RequestSource(str, Enum):
    DISCORD_MESSAGE = "discord_message"
    SLASH_COMMAND = "slash_command"
    INTERNAL = "internal"


class TrustLevel(str, Enum):
    SYSTEM = "system"
    APPLICATION = "application"
    USER = "user"
    EXTERNAL_DATA = "external_data"


@dataclass(frozen=True)
class PolicyRequest:
    content: str
    user_id: str
    guild_id: str | None = None
    channel_id: str | None = None
    source: RequestSource = RequestSource.DISCORD_MESSAGE
    mode: str = "balanced"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    content: str
    system_prompt: str | None = None
    reason: str | None = None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyResponse:
    content: str
    allowed: bool = True
    flags: tuple[str, ...] = ()
