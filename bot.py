import discord
from discord.ext import commands

from config.settings import GEMINI_API_KEY, DISCORD_TOKEN, validate_settings
from ai.router import AIRouter


validate_settings()


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
)

ai = AIRouter(GEMINI_API_KEY)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    print("HexxaAI is online!")


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    if message.content.lower() == "!ping":
        await message.channel.send("Pong! 🤖")
        return

    if bot.user in message.mentions:
        prompt = message.content.replace(
            f"<@{bot.user.id}>",
            "",
        ).strip()

        if not prompt:
            await message.channel.send(
                "You mentioned me! What do you want to talk about? 🤖"
            )
            return

        try:
            response = await ai.generate(prompt)
            await message.channel.send(response)

        except Exception as error:
            print(f"AI error: {error}")

            await message.channel.send(
                "Something went wrong while contacting my AI provider. 😭"
            )

    await bot.process_commands(message)


bot.run(DISCORD_TOKEN)