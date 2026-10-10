import asyncio

import cohere

from ai.providers.base import AIProvider, AIProviderError


class CohereProvider(AIProvider):
    name = "Cohere"

    DEFAULT_MODEL = "tiny-aya-global"
    TIMEOUT_SECONDS = 30

    def __init__(self, api_key: str):
        self.client = cohere.ClientV2(
            api_key=api_key,
            client_name="HexxaAI",
        )

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

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.chat,
                    model=selected_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=300,
                ),
                timeout=self.TIMEOUT_SECONDS,
            )

        except asyncio.TimeoutError as error:
            raise AIProviderError(
                "timeout",
                f"Cohere model '{selected_model}' took too long to respond.",
            ) from error

        except Exception as error:
            message = str(error)
            lowered = message.lower()

            if "429" in message or "rate limit" in lowered:
                category = "rate_limit"
                safe_message = "Cohere rate limit reached."

            elif "401" in message or "403" in message:
                category = "authentication"
                safe_message = "Cohere authentication failed."

            elif "400" in message:
                category = "invalid_request"
                safe_message = "Cohere rejected the request."

            elif (
                "connection" in lowered
                or "timeout" in lowered
                or "network" in lowered
            ):
                category = "network"
                safe_message = (
                    "A network error occurred while contacting Cohere."
                )

            else:
                category = "provider"
                safe_message = "Cohere returned an API error."

            raise AIProviderError(
                category,
                safe_message,
            ) from error

        # Cohere v2 response structure:
        # response.message.content[0].text
        try:
            response_text = response.message.content[0].text

        except (AttributeError, IndexError, TypeError) as error:
            raise AIProviderError(
                "empty_response",
                "Cohere returned no usable text content.",
            ) from error

        if not response_text or not response_text.strip():
            raise AIProviderError(
                "empty_response",
                "Cohere returned an empty response.",
            )

        return response_text.strip()

    async def close(self) -> None:
        return None
