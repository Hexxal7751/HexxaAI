import argparse
import statistics
import time

import cohere
from dotenv import load_dotenv
import os


load_dotenv()

PROMPT = "What is 25 × 37? Reply with only the answer."
EXPECTED_ANSWER = "925"

CANDIDATES = [
    "tiny-aya-global",
    "tiny-aya-earth",
    "tiny-aya-fire",
    "tiny-aya-water",
]


def request(client: cohere.ClientV2, model: str):
    start = time.perf_counter()
    first_token = None
    content = ""

    response = client.chat_stream(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        temperature=0,
        max_tokens=16,
    )

    for event in response:
        if event.type != "content-delta":
            continue

        text = event.delta.message.content.text

        if not text:
            continue

        if first_token is None:
            first_token = time.perf_counter() - start

        content += text

    total = time.perf_counter() - start

    return first_token, total, content.strip()


def benchmark(client: cohere.ClientV2, model: str, runs: int):
    print(f"\n  {model}")
    print("  Warming up...", end=" ", flush=True)

    try:
        request(client, model)
        print("OK")
    except Exception as error:
        print(f"FAILED: {error}")
        return None

    results = []

    for run in range(1, runs + 1):
        try:
            first, total, content = request(client, model)

            correct = content == EXPECTED_ANSWER

            results.append(
                {
                    "first": first,
                    "total": total,
                    "content": content,
                    "correct": correct,
                }
            )

            ttft = (
                f"{first:.3f}s"
                if first is not None
                else "N/A"
            )

            print(
                f"  Run {run}: "
                f"TTFT {ttft} | "
                f"total {total:.3f}s | "
                f"answer={content!r} | "
                f"{'CORRECT' if correct else 'WRONG'}"
            )

        except Exception as error:
            print(
                f"  Run {run}: "
                f"ERROR | {error}"
            )

    if not results:
        return None

    totals = [result["total"] for result in results]
    firsts = [
        result["first"]
        for result in results
        if result["first"] is not None
    ]

    return {
        "model": model,
        "runs": len(results),
        "correct": sum(
            result["correct"]
            for result in results
        ),
        "avg_total": statistics.mean(totals),
        "median_total": statistics.median(totals),
        "best_total": min(totals),
        "avg_first": (
            statistics.mean(firsts)
            if firsts
            else None
        ),
        "median_first": (
            statistics.median(firsts)
            if firsts
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Cohere Tiny Aya models "
            "for HexxaAI FAST."
        )
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured runs per model.",
    )

    args = parser.parse_args()

    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "COHERE_API_KEY is missing from .env"
        )

    client = cohere.ClientV2(
        api_key=api_key,
        client_name="HexxaAI FAST Benchmark",
    )

    print("=" * 78)
    print("HEXXAAI FAST — COHERE TINY AYA BENCHMARK")
    print("=" * 78)
    print(f"Prompt: {PROMPT!r}")
    print(f"Candidates: {len(CANDIDATES)}")
    print(f"Measured runs per model: {args.runs}")
    print("Each model receives one warm-up request first.")

    results = []

    for model in CANDIDATES:
        result = benchmark(
            client,
            model,
            args.runs,
        )

        if result:
            results.append(result)

    print("\n\n" + "=" * 78)
    print("FINAL RANKING — MEDIAN TOTAL RESPONSE TIME")
    print("=" * 78)

    ranked = sorted(
        results,
        key=lambda result: result["median_total"],
    )

    for index, result in enumerate(ranked, 1):
        print(
            f"{index:2}. "
            f"{result['model']:<22} "
            f"median={result['median_total']:.3f}s  "
            f"avg={result['avg_total']:.3f}s  "
            f"best={result['best_total']:.3f}s  "
            f"correct={result['correct']}/{result['runs']}"
        )

    print("\n" + "=" * 78)
    print("FINAL RANKING — MEDIAN TIME TO FIRST TOKEN")
    print("=" * 78)

    ranked_ttft = sorted(
        [
            result
            for result in results
            if result["median_first"] is not None
        ],
        key=lambda result: result["median_first"],
    )

    for index, result in enumerate(ranked_ttft, 1):
        print(
            f"{index:2}. "
            f"{result['model']:<22} "
            f"median TTFT="
            f"{result['median_first']:.3f}s"
        )

    print("\n" + "=" * 78)
    print("BENCHMARK COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
