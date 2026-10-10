import asyncio
import logging
import time

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from ai import AIMode, AIService, AIRouter
from ai.providers.base import AIProviderError
from ai.providers.cohere import CohereProvider
from ai.providers.gemini import GeminiProvider
from config.settings import (
    COHERE_API_KEY,
    DISCORD_TOKEN,
    GEMINI_API_KEY,
    validate_settings,
)
from policy import PolicyEngine


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DISCORD_MESSAGE_LIMIT = 2000

ALLOWED_MENTIONS = discord.AllowedMentions(
    users=True,
    roles=False,
    everyone=False,
)

GENERIC_ERROR_MESSAGE = (
    "Something went wrong while processing your request. 😭"
)


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("hexxaai")

validate_settings()


# --------------------------------------------------
# Discord bot setup
# --------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
)


# --------------------------------------------------
# AI provider setup
# --------------------------------------------------

gemini = GeminiProvider(GEMINI_API_KEY)
cohere = CohereProvider(COHERE_API_KEY)

router = AIRouter(
    {
        "gemini": gemini,
        "cohere": cohere,
    }
)

ai = AIService(
    router,
    policy=PolicyEngine(),
)


# --------------------------------------------------
# Discord HTTP diagnostics
# --------------------------------------------------

def configure_http_diagnostics() -> None:
    """
    Trace Discord HTTP request timings.

    Logs HTTP methods, hostnames, status codes, and elapsed
    times. Does not log request URLs, tokens, or message content.
    """

    trace = aiohttp.TraceConfig()

    async def request_start(session, context, params):
        context.started_at = time.perf_counter()

        logger.info(
            "HTTP request started: method=%s host=%s",
            params.method,
            params.url.host,
        )

    async def request_end(session, context, params):
        started_at = getattr(context, "started_at", None)

        elapsed = (
            time.perf_counter() - started_at
            if started_at is not None
            else 0.0
        )

        logger.info(
            "HTTP response received: method=%s host=%s "
            "status=%s elapsed=%.3fs",
            params.method,
            params.url.host,
            params.response.status,
            elapsed,
        )

    async def request_exception(session, context, params):
        started_at = getattr(context, "started_at", None)

        elapsed = (
            time.perf_counter() - started_at
            if started_at is not None
            else 0.0
        )

        logger.warning(
            "HTTP request failed: method=%s host=%s elapsed=%.3fs",
            params.method,
            params.url.host,
            elapsed,
        )

    async def connection_start(session, context, params):
        logger.info("HTTP connection setup started")

    async def connection_end(session, context, params):
        logger.info("HTTP connection setup completed")

    async def dns_start(session, context, params):
        logger.info("HTTP DNS lookup started")

    async def dns_end(session, context, params):
        logger.info("HTTP DNS lookup completed")

    trace.on_request_start.append(request_start)
    trace.on_request_end.append(request_end)
    trace.on_request_exception.append(request_exception)

    trace.on_connection_create_start.append(connection_start)
    trace.on_connection_create_end.append(connection_end)

    trace.on_dns_resolvehost_start.append(dns_start)
    trace.on_dns_resolvehost_end.append(dns_end)

    # discord.py creates its HTTP session during login.
    # Configure tracing before bot.start() creates that session.
    bot.http.http_trace = trace

    logger.info("Discord HTTP diagnostics enabled")


# --------------------------------------------------
# Message helpers
# --------------------------------------------------

def extract_mention_prompt(message: discord.Message) -> str:
    """Remove bot mentions from a message and return its prompt."""

    if bot.user is None:
        return message.content.strip()

    prompt = message.content.replace(
        f"<@{bot.user.id}>",
        "",
    )

    prompt = prompt.replace(
        f"<@!{bot.user.id}>",
        "",
    )

    return prompt.strip()


def split_response(
    text: str,
    limit: int = DISCORD_MESSAGE_LIMIT,
) -> list[str]:
    """Split a response into Discord-safe message chunks."""

    if not text:
        return ["I couldn't generate a response. 😭"]

    chunks = []

    while len(text) > limit:
        split_at = text.rfind("\n", 0, limit)

        if split_at <= 0:
            split_at = text.rfind(" ", 0, limit)

        if split_at <= 0:
            split_at = limit

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()

    if text:
        chunks.append(text)

    return chunks


async def send_response(
    message: discord.Message,
    response: str,
) -> None:
    """Send an AI response while respecting Discord's message limit."""

    chunks = split_response(response)

    try:
        await message.reply(
            chunks[0],
            mention_author=False,
            allowed_mentions=ALLOWED_MENTIONS,
        )
    except discord.HTTPException:
        logger.warning(
            "Reply failed; attempting a normal channel message",
            exc_info=True,
        )

        await message.channel.send(
            chunks[0],
            allowed_mentions=ALLOWED_MENTIONS,
        )

    for chunk in chunks[1:]:
        await message.channel.send(
            chunk,
            allowed_mentions=ALLOWED_MENTIONS,
        )


# --------------------------------------------------
# Slash-command helpers
# --------------------------------------------------

def interaction_kwargs(
    interaction: discord.Interaction,
) -> dict:
    """Build request metadata for the AI service."""

    return {
        "user_id": str(interaction.user.id),
        "channel_id": str(interaction.channel_id),
        "guild_id": (
            str(interaction.guild_id)
            if interaction.guild_id is not None
            else None
        ),
        "source": "slash_command",
    }


async def acknowledge_interaction(
    interaction: discord.Interaction,
) -> bool:
    """
    Acknowledge a slash command before performing slow AI work.

    Discord requires the initial interaction response within
    approximately three seconds. Deferring does not extend that
    initial deadline; it acknowledges the interaction and permits
    a later response through followups.
    """

    command_name = (
        interaction.command.name
        if interaction.command is not None
        else "unknown"
    )

    age = (
        discord.utils.utcnow() - interaction.created_at
    ).total_seconds()

    logger.info(
        "Slash command received: command=%s interaction=%s age=%.3fs",
        command_name,
        interaction.id,
        age,
    )

    started_at = time.perf_counter()

    try:
        await interaction.response.defer(thinking=True)

    except discord.NotFound:
        elapsed = time.perf_counter() - started_at

        logger.error(
            "Discord rejected the initial acknowledgement: "
            "command=%s interaction=%s age=%.3fs "
            "ack_request_elapsed=%.3fs. "
            "The interaction may have expired.",
            command_name,
            interaction.id,
            (
                discord.utils.utcnow() - interaction.created_at
            ).total_seconds(),
            elapsed,
            exc_info=True,
        )

        return False

    except discord.HTTPException:
        elapsed = time.perf_counter() - started_at

        logger.error(
            "HTTP error acknowledging slash command: "
            "command=%s interaction=%s "
            "ack_request_elapsed=%.3fs",
            command_name,
            interaction.id,
            elapsed,
            exc_info=True,
        )

        return False

    except discord.DiscordException:
        logger.exception(
            "Discord error acknowledging slash command: "
            "command=%s interaction=%s",
            command_name,
            interaction.id,
        )

        return False

    elapsed = time.perf_counter() - started_at

    logger.info(
        "Slash command acknowledged: command=%s "
        "interaction=%s ack_request_elapsed=%.3fs",
        command_name,
        interaction.id,
        elapsed,
    )

    return True


async def send_interaction_response(
    interaction: discord.Interaction,
    response: str,
) -> None:
    """Send an AI response using Discord interaction followups."""

    chunks = split_response(response)

    await interaction.followup.send(
        f"{interaction.user.mention} {chunks[0]}",
        allowed_mentions=ALLOWED_MENTIONS,
    )

    for chunk in chunks[1:]:
        await interaction.followup.send(
            chunk,
            allowed_mentions=ALLOWED_MENTIONS,
        )


async def report_interaction_error(
    interaction: discord.Interaction,
    message: str = GENERIC_ERROR_MESSAGE,
) -> None:
    """Report an error without sending a second initial response."""

    try:
        if interaction.response.is_done():
            await interaction.followup.send(
                f"{interaction.user.mention} {message}",
                allowed_mentions=ALLOWED_MENTIONS,
            )
        else:
            await interaction.response.send_message(
                message,
                ephemeral=True,
                allowed_mentions=ALLOWED_MENTIONS,
            )

    except discord.DiscordException:
        logger.exception(
            "Could not report interaction error: interaction=%s",
            interaction.id,
        )


async def generate_for_interaction(
    interaction: discord.Interaction,
    prompt: str,
    *,
    mode: AIMode | None = None,
    model_name: str | None = None,
) -> str:
    """Generate an AI response for a slash command."""

    kwargs = interaction_kwargs(interaction)

    if model_name is not None:
        return await ai.generate_named_model(
            prompt,
            model_name=model_name,
            **kwargs,
        )

    return await ai.generate(
        prompt,
        mode=mode or AIMode.BALANCED,
        **kwargs,
    )


async def run_slash_command(
    interaction: discord.Interaction,
    prompt: str,
    *,
    mode: AIMode | None = None,
    model_name: str | None = None,
) -> None:
    """
    Handle slash commands.

    The acknowledgement must finish before AI generation begins.
    """

    acknowledged = await acknowledge_interaction(interaction)

    if not acknowledged:
        logger.warning(
            "Skipping AI generation because the interaction "
            "was not acknowledged: interaction=%s",
            interaction.id,
        )
        return

    try:
        response = await generate_for_interaction(
            interaction,
            prompt,
            mode=mode,
            model_name=model_name,
        )

        await send_interaction_response(
            interaction,
            response,
        )

    except AIProviderError:
        logger.exception(
            "AI provider error: interaction=%s",
            interaction.id,
        )

        await report_interaction_error(
            interaction,
            "My AI provider is having trouble right now. Try again shortly. 😭",
        )

    except discord.NotFound:
        logger.warning(
            "Interaction followup was rejected or expired: "
            "interaction=%s",
            interaction.id,
            exc_info=True,
        )

    except discord.DiscordException:
        logger.exception(
            "Discord error processing slash command: interaction=%s",
            interaction.id,
        )

        await report_interaction_error(interaction)

    except Exception:
        logger.exception(
            "Unexpected slash-command error: interaction=%s",
            interaction.id,
        )

        await report_interaction_error(interaction)


# --------------------------------------------------
# Slash commands
# --------------------------------------------------

@bot.tree.command(
    name="ask",
    description="Ask HexxaAI anything.",
)
@app_commands.describe(
    prompt="What would you like to ask?",
)
async def ask(
    interaction: discord.Interaction,
    prompt: str,
) -> None:
    """Ask HexxaAI using the balanced AI mode."""

    await run_slash_command(
        interaction,
        prompt,
        mode=AIMode.BALANCED,
    )


@bot.tree.command(
    name="model",
    description="Choose which AI model HexxaAI uses.",
)
@app_commands.describe(
    prompt="What would you like to ask?",
    model="Choose your AI model.",
)
@app_commands.choices(
    model=[
        app_commands.Choice(
            name="Tiny Aya — Fast",
            value="tiny-aya",
        ),
        app_commands.Choice(
            name="Gemini Flash — Balanced",
            value="gemini-flash",
        ),
    ],
)
async def model(
    interaction: discord.Interaction,
    model: app_commands.Choice[str],
    prompt: str,
) -> None:
    """Ask HexxaAI using a selected model."""

    await run_slash_command(
        interaction,
        prompt,
        model_name=model.value,
    )


# --------------------------------------------------
# Discord events
# --------------------------------------------------

@bot.event
async def on_ready() -> None:
    """Log in and synchronize application commands."""

    logger.info("Logged in as %s", bot.user)
    logger.info("HexxaAI is online!")
    logger.info("AI providers: Cohere, Gemini")

    try:
        synced = await bot.tree.sync()

        logger.info(
            "Synced %s application command(s).",
            len(synced),
        )

    except discord.DiscordException:
        logger.exception("Failed to synchronize application commands")


@bot.event
async def on_message(message: discord.Message) -> None:
    """Handle mentions and regular prefix commands."""

    if message.author.bot:
        return

    if bot.user and bot.user.mentioned_in(message):
        prompt = extract_mention_prompt(message)

        if not prompt:
            await message.reply(
                "Hey! Mention me with a question and I'll help. 😎",
                mention_author=False,
                allowed_mentions=ALLOWED_MENTIONS,
            )
            return

        try:
            response = await ai.generate(
                prompt,
                mode=AIMode.FAST,
                user_id=str(message.author.id),
                channel_id=str(message.channel.id),
                guild_id=(
                    str(message.guild.id)
                    if message.guild is not None
                    else None
                ),
                source="mention",
            )

            await send_response(message, response)

        except AIProviderError:
            logger.exception("AI provider error while handling a mention")

            await message.reply(
                GENERIC_ERROR_MESSAGE,
                mention_author=False,
                allowed_mentions=ALLOWED_MENTIONS,
            )

        except Exception:
            logger.exception("Unexpected error while handling a mention")

            await message.reply(
                GENERIC_ERROR_MESSAGE,
                mention_author=False,
                allowed_mentions=ALLOWED_MENTIONS,
            )

        return

    await bot.process_commands(message)


# --------------------------------------------------
# Application lifecycle
# --------------------------------------------------

async def main() -> None:
    """Start HexxaAI and clean up its AI providers."""

    configure_http_diagnostics()

    try:
        async with bot:
            await bot.start(DISCORD_TOKEN)

    finally:
        await ai.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("HexxaAI stopped by the user.")