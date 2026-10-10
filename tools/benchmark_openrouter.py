import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT = "What is 25 × 37? Reply with only the answer."
EXPECTED_ANSWER = "925"

CANDIDATES = [
    "liquid/lfm-2.5-2.6b:free",
    "apodex/apodex-1.1-mini:free",
    "thinkingmachines/inkling-small:free",
    "poolside/laguna-xs-2.1:free",
    "inclusionai/ling-3.0-flash-sante:free",
    "inclusionai/ling-3.1-flash",
    "nvidia/nemotron-3.5-lightning:free",
    "thinkingmachines/inkling:free",
]


def request(model: str):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        "temperature": 0,
        "max_tokens": 16,
        "stream": True,
    }

    request_obj = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Hexxal7751/HexxaAI",
            "X-Title": "HexxaAI FAST Benchmark",
        },
        method="POST",
    )

    start = time.perf_counter()
    first_token = None
    content = ""

    with urllib.request.urlopen(request_obj, timeout=30) as response:
        for raw_line in response:
            line = raw_line.decode(errors="ignore").strip()

            if not line.startswith("data: "):
                continue

            data = line[6:]

            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            delta = (
                chunk.get("choices", [{}])[0]
                .get("delta", {})
                .get("content")
            )

            if delta:
                if first_token is None:
                    first_token = time.perf_counter() - start

                content += delta

    total = time.perf_counter() - start

    return first_token, total, content.strip()


def benchmark(model: str, runs: int):
    print(f"\n  {model}")
    print("  Warming up...", end=" ", flush=True)

    try:
        request(model)
        print("OK")
    except Exception as error:
        print(f"FAILED: {error}")
        return None

    results = []

    for run in range(1, runs + 1):
        try:
            first, total, content = request(model)

            correct = content.strip() == EXPECTED_ANSWER

            results.append(
                {
                    "first": first,
                    "total": total,
                    "content": content,
                    "correct": correct,
                }
            )

            ttft = f"{first:.3f}s" if first is not None else "N/A"

            print(
                f"  Run {run}: "
                f"TTFT {ttft} | "
                f"total {total:.3f}s | "
                f"answer={content!r} | "
                f"{'CORRECT' if correct else 'WRONG'}"
            )

        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="ignore")
            print(
                f"  Run {run}: HTTP {error.code} | "
                f"{body[:300]}"
            )

        except Exception as error:
            print(f"  Run {run}: ERROR | {error}")

    if not results:
        return None

    totals = [r["total"] for r in results]
    firsts = [r["first"] for r in results if r["first"] is not None]

    return {
        "model": model,
        "runs": len(results),
        "correct": sum(r["correct"] for r in results),
        "avg_total": statistics.mean(totals),
        "median_total": statistics.median(totals),
        "best_total": min(totals),
        "avg_first": statistics.mean(firsts) if firsts else None,
        "median_first": statistics.median(firsts) if firsts else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OpenRouter free models for HexxaAI FAST."
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured runs per model.",
    )

    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing from .env"
        )

    print("=" * 78)
    print("HEXXAAI FAST — OPENROUTER BENCHMARK")
    print("=" * 78)
    print(f"Prompt: {PROMPT!r}")
    print(f"Candidates: {len(CANDIDATES)}")
    print(f"Measured runs per model: {args.runs}")
    print("Each model receives one warm-up request first.")

    results = []

    for model in CANDIDATES:
        result = benchmark(model, args.runs)

        if result:
            results.append(result)

    print("\n\n" + "=" * 78)
    print("FINAL RANKING — MEDIAN TOTAL RESPONSE TIME")
    print("=" * 78)

    ranked = sorted(results, key=lambda x: x["median_total"])

    for index, result in enumerate(ranked, 1):
        print(
            f"{index:2}. "
            f"{result['model']:<48} "
            f"median={result['median_total']:.3f}s  "
            f"avg={result['avg_total']:.3f}s  "
            f"best={result['best_total']:.3f}s  "
            f"correct={result['correct']}/{result['runs']}"
        )

    print("\n" + "=" * 78)
    print("FINAL RANKING — MEDIAN TIME TO FIRST TOKEN")
    print("=" * 78)

    ranked_ttft = sorted(
        [r for r in results if r["median_first"] is not None],
        key=lambda x: x["median_first"],
    )

    for index, result in enumerate(ranked_ttft, 1):
        print(
            f"{index:2}. "
            f"{result['model']:<48} "
            f"median TTFT={result['median_first']:.3f}s"
        )

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
