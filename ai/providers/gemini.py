from google import genai


class GeminiProvider:
    """Production Gemini AI provider."""

    MODEL = "gemini-3.8-flash"

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    async def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.MODEL,
            contents=prompt,
        )

        return response.text