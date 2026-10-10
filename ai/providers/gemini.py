import asyncio

import httpx
from google import genai
from google.genai import errors as genai_errors

from ai.providers.base import AIProvider, AIProviderError


class GeminiProvider(AIProvider):
    """Gemini implementation of the HexxaAI AI provider interface."""

    name = "Gemini"

    DEFAULT_MODEL = "gemini-3-flash-preview"
    TIMEOUT_SECONDS = 60

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key).aio

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        prompt = prompt.strip()

        if not prompt:
            raise AIProviderError(
                "invalid_request",
                "The AI prompt was empty.",
            )

        selected_model = model or self.DEFAULT_MODEL

        config = {
            "temperature": 0.7,
        }

        if system_prompt:
            config["system_instruction"] = system_prompt

        try:
            response = await asyncio.wait_for(
                self.client.models.generate_content(
                    model=selected_model,
                    contents=prompt,
                    config=config,
                ),
                timeout=self.TIMEOUT_SECONDS,
            )

        except asyncio.TimeoutError as error:
            raise AIProviderError(
                "timeout",
                f"Gemini model '{selected_model}' took too long to respond.",
            ) from error

        except genai_errors.APIError as error:
            status_code = getattr(error, "code", None)

            if status_code in (401, 403):
                category = "authentication"
                message = "Gemini authentication failed."
            elif status_code == 429:
                category = "rate_limit"
                message = "Gemini rate limit reached."
            elif status_code == 400:
                category = "invalid_request"
                message = "Gemini rejected the request."
            elif status_code is not None and status_code >= 500:
                category = "service"
                message = "Gemini is temporarily unavailable."
            else:
                category = "provider"
                message = "Gemini returned an API error."

            raise AIProviderError(category, message) from error

        except (httpx.HTTPError, ConnectionError, OSError) as error:
            raise AIProviderError(
                "network",
                "A network error occurred while contacting Gemini.",
            ) from error

        except Exception as error:
            raise AIProviderError(
                "provider",
                "An unexpected Gemini error occurred.",
            ) from error

        response_text = getattr(response, "text", None)

        if not response_text or not response_text.strip():
            raise AIProviderError(
                "empty_response",
                "Gemini returned an empty response.",
            )

        return response_text.strip()

    async def close(self) -> None:
        await self.client.aclose()
