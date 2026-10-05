from abc import ABC, abstractmethod


class AIProviderError(Exception):
    """Controlled error raised by an AI provider."""

    def __init__(self, category: str, message: str):
        self.category = category
        self.message = message

        super().__init__(message)


class AIProvider(ABC):
    """Base interface for all HexxaAI AI providers."""

    name = "unknown"

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a response from a prompt."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release provider resources."""
        return None