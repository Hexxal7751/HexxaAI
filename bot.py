import os

import discord
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")

if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN is not set.")


intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    print("HexxaAI is online!")


@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.content.lower() == "!ping":
        await message.channel.send("Pong! 🤖")


client.run(TOKEN)