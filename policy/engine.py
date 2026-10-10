from policy.input import validate_input
from policy.models import PolicyDecision, PolicyRequest, PolicyResponse
from policy.output import validate_output
from policy.prompt import HEXXAAI_SYSTEM_PROMPT, build_user_content


class PolicyEngine:
    """
    Universal application-level security and policy boundary.

    Providers never decide application trust or permissions.
    """

    def prepare_request(
        self,
        request: PolicyRequest,
    ) -> PolicyDecision:
        decision = validate_input(request)

        if not decision.allowed:
            return decision

        return PolicyDecision(
            allowed=True,
            content=build_user_content(
                request,
                decision.content,
            ),
            system_prompt=HEXXAAI_SYSTEM_PROMPT,
            flags=decision.flags,
        )

    def validate_response(
        self,
        content: str,
    ) -> PolicyResponse:
        return validate_output(content)
