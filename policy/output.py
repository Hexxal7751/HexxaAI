from policy.models import PolicyResponse

MAX_OUTPUT_LENGTH = 12_000

IDENTITY_LEAK_PATTERNS = (
    "i'm aya",
    "i am aya",
    "i’m aya",
    "aya, a friendly language model",
    "aya, a friendly discord companion",
    "language model from cohere",
    "language model built by cohere",
    "ai model from cohere",
    "ai model built by cohere",
)


def validate_output(content: str) -> PolicyResponse:
    content = content.strip()

    if not content:
        return PolicyResponse(
            content="",
            allowed=False,
            flags=("empty_output",),
        )

    lowered = content.lower()

    if any(pattern in lowered for pattern in IDENTITY_LEAK_PATTERNS):
        return PolicyResponse(
            content=(
                "I'm HexxaAI, your Discord AI companion. "
                "I'm built to chat, help, and occasionally cause problems. 😎"
            ),
            allowed=True,
            flags=("identity_leak_blocked",),
        )

    if len(content) > MAX_OUTPUT_LENGTH:
        return PolicyResponse(
            content=content[:MAX_OUTPUT_LENGTH].rstrip(),
            allowed=True,
            flags=("output_length_limit",),
        )

    return PolicyResponse(
        content=content,
        allowed=True,
    )