from ai.providers.base import AIProvider


class AIRouter:
    """Routes AI requests through the configured provider."""

    def __init__(self, provider: AIProvider):
        self.provider = provider

    async def generate(self, prompt: str) -> str:
        """Generate a response using the active AI provider."""

        return await self.provider.generate(prompt)

    async def close(self) -> None:
        """Close the active AI provider."""

        await self.provider.close()

    @property
    def provider_name(self) -> str:
        """Return the active provider name."""

        return self.provider.name