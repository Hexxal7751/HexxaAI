from ai.providers.gemini import GeminiProvider


class AIRouter:
    """Routes AI requests to the configured provider."""

    def __init__(self, gemini_api_key: str):
        self.gemini = GeminiProvider(gemini_api_key)

    async def generate(self, prompt: str) -> str:
        return await self.gemini.generate(prompt)