from enum import Enum

from ai.providers.base import AIProvider


class AIMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    REASONING = "reasoning"


class AIRouter:
    MODELS = {
        AIMode.FAST: "tiny-aya-global",
        AIMode.BALANCED: "gemini-3-flash-preview",
        AIMode.REASONING: "gemini-3.1-pro-preview",
    }

    PROVIDERS = {
        AIMode.FAST: "cohere",
        AIMode.BALANCED: "gemini",
        AIMode.REASONING: "gemini",
    }

    DEFAULT_MODE = AIMode.BALANCED

    def __init__(self, providers: dict[str, AIProvider]):
        self.providers = providers

    @staticmethod
    def normalize_mode(mode: AIMode | str) -> AIMode:
        try:
            return (
                mode
                if isinstance(mode, AIMode)
                else AIMode(mode.lower())
            )
        except (AttributeError, ValueError) as error:
            raise ValueError(f"Unknown AI mode: {mode}") from error

    async def generate(
        self,
        prompt: str,
        mode: AIMode | str = DEFAULT_MODE,
        system_prompt: str | None = None,
    ) -> str:
        selected_mode = self.normalize_mode(mode)

        return await self.generate_model(
            prompt,
            provider=self.PROVIDERS[selected_mode],
            model=self.MODELS[selected_mode],
            system_prompt=system_prompt,
        )

    async def generate_model(
        self,
        prompt: str,
        provider: str,
        model: str,
        system_prompt: str | None = None,
    ) -> str:
        selected_provider = self.providers.get(provider)

        if selected_provider is None:
            raise RuntimeError(
                f"AI provider '{provider}' is not configured."
            )

        return await selected_provider.generate(
            prompt,
            model=model,
            system_prompt=system_prompt,
        )

    async def close(self) -> None:
        closed = set()

        for provider in self.providers.values():
            identity = id(provider)

            if identity in closed:
                continue

            closed.add(identity)
            await provider.close()

    @property
    def provider_name(self) -> str:
        providers = {
            provider.name
            for provider in self.providers.values()
        }

        return ", ".join(sorted(providers))

    @classmethod
    def model_for_mode(
        cls,
        mode: AIMode | str,
    ) -> str:
        return cls.MODELS[cls.normalize_mode(mode)]

    @classmethod
    def provider_for_mode(
        cls,
        mode: AIMode | str,
    ) -> str:
        return cls.PROVIDERS[cls.normalize_mode(mode)]
