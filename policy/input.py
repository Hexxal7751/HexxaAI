from policy.models import PolicyDecision, PolicyRequest


MAX_INPUT_LENGTH = 12_000


def validate_input(request: PolicyRequest) -> PolicyDecision:
    content = request.content.strip()

    if not content:
        return PolicyDecision(
            allowed=False,
            content="",
            reason="empty_input",
        )

    if len(content) > MAX_INPUT_LENGTH:
        return PolicyDecision(
            allowed=False,
            content="",
            reason="input_too_large",
            flags=("input_length_limit",),
        )

    return PolicyDecision(
        allowed=True,
        content=content,
    )
