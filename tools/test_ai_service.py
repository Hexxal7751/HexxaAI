import asyncio

from ai import AIMode
from bot import ai


async def main():
    print("=" * 70)
    print("HEXXAAI AI SERVICE INTEGRATION TEST")
    print("=" * 70)

    try:
        print("\nTesting FAST → Cohere → Tiny Aya...")
        response = await ai.generate(
            "Reply with exactly: POLICY_OK",
            mode=AIMode.FAST,
            user_id="integration-test",
        )

        print(f"Response: {response!r}")

        if not response:
            raise RuntimeError("Empty response.")

        print("PASS: FAST pipeline returned a response.")

    finally:
        await ai.close()

    print("\n" + "=" * 70)
    print("AI SERVICE INTEGRATION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
