from ai.router import AIMode, AIRouter
from policy import PolicyEngine, PolicyRequest


class AIService:
    def __init__(
        self,
        router: AIRouter,
        policy: PolicyEngine | None = None,
    ):
        self.router = router
        self.policy = policy or PolicyEngine()

    async def generate(
        self,
        prompt,
        mode=AIMode.BALANCED,
        *,
        user_id="unknown",
        guild_id=None,
        channel_id=None,
        source="discord_message",
    ):
        request = PolicyRequest(
            content=prompt,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            mode=(
                mode.value
                if isinstance(mode, AIMode)
                else mode
            ),
            metadata={"source": source},
        )

        decision = self.policy.prepare_request(request)

        if not decision.allowed:
            raise ValueError(
                decision.reason or "Request rejected by policy."
            )

        response = await self.router.generate(
            decision.content,
            mode=mode,
            system_prompt=decision.system_prompt,
        )

        return self._validate_response(response)

    async def generate_named_model(
        self,
        prompt,
        name,
        *,
        user_id="unknown",
        guild_id=None,
        channel_id=None,
        source="slash_command",
    ):
        request = PolicyRequest(
            content=prompt,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            source=source,
            metadata={"model": name},
        )

        decision = self.policy.prepare_request(request)

        if not decision.allowed:
            raise ValueError(
                decision.reason or "Request rejected by policy."
            )

        response = await self.router.generate_named_model(
            decision.content,
            name=name,
            system_prompt=decision.system_prompt,
        )

        return self._validate_response(response)

    async def generate_model(
        self,
        prompt,
        provider,
        model,
        *,
        user_id="unknown",
        guild_id=None,
        channel_id=None,
        source="slash_command",
    ):
        request = PolicyRequest(
            content=prompt,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            source=source,
            metadata={
                "provider": provider,
                "model": model,
            },
        )

        decision = self.policy.prepare_request(request)

        if not decision.allowed:
            raise ValueError(
                decision.reason or "Request rejected by policy."
            )

        response = await self.router.generate_model(
            decision.content,
            provider=provider,
            model=model,
            system_prompt=decision.system_prompt,
        )

        return self._validate_response(response)

    def _validate_response(self, response: str) -> str:
        validated = self.policy.validate_response(response)

        if not validated.allowed:
            raise ValueError("AI response rejected by policy.")

        return validated.content

    async def close(self):
        await self.router.close()

    @property
    def provider_name(self):
        return self.router.provider_name
