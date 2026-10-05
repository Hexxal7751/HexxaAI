import asyncio
import logging

import discord
from discord.ext import commands

from ai.providers.base import AIProviderError
from ai.providers.gemini import GeminiProvider
from ai.router import AIRouter
from config.settings import (
    DISCORD_TOKEN,
    GEMINI_API_KEY,
    validate_settings,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DISCORD_MESSAGE_LIMIT = 2000


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("hexxaai")


# --------------------------------------------------
# Environment validation
# --------------------------------------------------

validate_settings()


# --------------------------------------------------
# Discord bot
# --------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
)


# --------------------------------------------------
# AI system
# --------------------------------------------------

gemini = GeminiProvider(GEMINI_API_KEY)
ai = AIRouter(gemini)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def extract_mention_prompt(message: discord.Message) -> str:
    """Remove HexxaAI's mention from a Discord message."""

    if bot.user is None:
        return ""

    content = message.content

    mention_formats = (
        f"<@{bot.user.id}>",
        f"<@!{bot.user.id}>",
    )

    for mention in mention_formats:
        content = content.replace(mention, "")

    return content.strip()


def split_response(
    text: str,
    limit: int = DISCORD_MESSAGE_LIMIT,
) -> list[str]:
    """Split a response into Discord-safe message chunks."""

    if len(text) <= limit:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit + 1)

        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit + 1)

        if split_at <= 0:
            split_at = limit

        chunk = remaining[:split_at].rstrip()

        if chunk:
            chunks.append(chunk)

        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)

    return chunks


async def send_response(
    channel: discord.abc.Messageable,
    response: str,
) -> None:
    """Send an AI response while respecting Discord's message limit."""

    for chunk in split_response(response):
        await channel.send(chunk)


# --------------------------------------------------
# Discord events
# --------------------------------------------------

@bot.event
async def on_ready():
    """Called when HexxaAI successfully connects to Discord."""

    logger.info("Logged in as %s", bot.user)
    logger.info("HexxaAI is online!")
    logger.info("AI provider: %s", ai.provider_name)


@bot.event
async def on_message(message: discord.Message):
    """Handle incoming Discord messages."""

    if message.author.bot:
        return

    # ------------------------------
    # Basic ping command
    # ------------------------------

    if message.content.lower().strip() == "!ping":
        await message.channel.send("Pong! 🤖")
        return

    # ------------------------------
    # AI mention handling
    # ------------------------------

    if bot.user not in message.mentions:
        await bot.process_commands(message)
        return

    prompt = extract_mention_prompt(message)

    if not prompt:
        await message.channel.send(
            "You mentioned me! What do you want to talk about? 🤖"
        )
        return

    # ------------------------------
    # AI generation
    # ------------------------------

    try:
        # Discord immediately displays the typing indicator
        # while Gemini is generating the response.
        async with message.channel.typing():
            response = await ai.generate(prompt)

        await send_response(
            message.channel,
            response,
        )

    except AIProviderError as error:
        logger.error(
            "AI provider error [%s]: %s",
            error.category,
            error.message,
        )

        await message.channel.send(
            "I ran into a problem while generating that response. 😭"
        )

    except discord.DiscordException:
        logger.exception(
            "Discord error while sending AI response."
        )

    except Exception:
        logger.exception(
            "Unexpected error while processing message."
        )

        await message.channel.send(
            "Something unexpected went wrong. 😭"
        )

    await bot.process_commands(message)


# --------------------------------------------------
# Application lifecycle
# --------------------------------------------------

async def main():
    """Start HexxaAI and cleanly release resources on shutdown."""

    try:
        await bot.start(DISCORD_TOKEN)

    finally:
        await ai.close()

        if not bot.is_closed():
            await bot.close()


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())