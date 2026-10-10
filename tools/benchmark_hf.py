import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/v1/chat/completions"

PROMPT = "What is 25 × 37? Reply with only the answer."

SHORTLIST = {
    "Tiny Aya": [
        "CohereLabs/tiny-aya-earth:cohere",
        "CohereLabs/tiny-aya-global:cohere",
        "CohereLabs/tiny-aya-water:cohere",
        "CohereLabs/tiny-aya-fire:cohere",
    ],
    "Ultra-fast infrastructure": [
        "openai/gpt-oss-20b:groq",
        "openai/gpt-oss-120b:groq",
        "openai/gpt-oss-120b:cerebras",
    ],
    "Other fast candidates": [
        "ibm-granite/granite-4.2-3b:deepinfra",
        "Qwen/Qwen3.5-9B:ovhcloud",
        "zai-org/GLM-5.3-Flash:together",
    ],
}


def request(model: str, stream: bool = True):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        "temperature": 0,
        "max_tokens": 32,
        "stream": stream,
    }

    data = json.dumps(payload).encode()

    request_obj = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    start = time.perf_counter()
    first_token = None
    content = ""

    with urllib.request.urlopen(request_obj, timeout=30) as response:
        if not stream:
            body = json.loads(response.read().decode())
            content = body["choices"][0]["message"].get("content", "")
            total = time.perf_counter() - start
            return first_token, total, content

        for raw_line in response:
            line = raw_line.decode(errors="ignore").strip()

            if not line.startswith("data: "):
                continue

            payload_text = line[6:]

            if payload_text == "[DONE]":
                break

            try:
                chunk = json.loads(payload_text)
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
    return first_token, total, content


def benchmark_model(model: str, runs: int):
    print(f"\n  {model}")
    print("  Warming up...", end=" ", flush=True)

    try:
        request(model, stream=True)
        print("OK")
    except Exception as error:
        print(f"FAILED: {error}")
        return None

    results = []

    for run in range(1, runs + 1):
        try:
            first, total, content = request(model, stream=True)

            result = {
                "first": first,
                "total": total,
                "content": content.strip(),
            }

            results.append(result)

            first_text = (
                f"{first:.3f}s" if first is not None else "N/A"
            )

            print(
                f"  Run {run}: "
                f"TTFT {first_text} | "
                f"total {total:.3f}s | "
                f"answer={content.strip()!r}"
            )

        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="ignore")
            print(f"  Run {run}: HTTP {error.code} | {body[:300]}")

        except Exception as error:
            print(f"  Run {run}: ERROR | {error}")

    if not results:
        return None

    totals = [r["total"] for r in results]
    firsts = [r["first"] for r in results if r["first"] is not None]

    return {
        "model": model,
        "runs": len(results),
        "avg_total": statistics.mean(totals),
        "median_total": statistics.median(totals),
        "best_total": min(totals),
        "avg_first": statistics.mean(firsts) if firsts else None,
        "median_first": statistics.median(firsts) if firsts else None,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HexxaAI FAST candidates through Hugging Face."
    )

    parser.add_argument(
        "--shortlist",
        action="store_true",
        help="Benchmark the curated FAST shortlist.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured runs per model.",
    )

    args = parser.parse_args()

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing from .env")

    if not args.shortlist:
        print("Use: python3 tools/benchmark_hf.py --shortlist --runs 3")
        return

    print("=" * 78)
    print("HEXXAAI FAST — REAL RESPONSE-TIME BENCHMARK")
    print("=" * 78)
    print(f"Prompt: {PROMPT!r}")
    print(f"Measured runs: {args.runs}")
    print("Each model receives one warm-up request first.")
    print()

    all_results = []

    for group_name, models in SHORTLIST.items():
        print("\n" + "=" * 78)
        print(group_name.upper())
        print("=" * 78)

        for model in models:
            result = benchmark_model(model, args.runs)

            if result:
                result["group"] = group_name
                all_results.append(result)

    print("\n\n" + "=" * 78)
    print("FINAL RANKING — TOTAL RESPONSE TIME")
    print("=" * 78)

    ranked = sorted(all_results, key=lambda x: x["median_total"])

    for index, result in enumerate(ranked, 1):
        print(
            f"{index:2}. "
            f"{result['model']:<48} "
            f"median={result['median_total']:.3f}s  "
            f"avg={result['avg_total']:.3f}s  "
            f"best={result['best_total']:.3f}s  "
            f"success={result['runs']}/{args.runs}"
        )

    print("\n" + "=" * 78)
    print("FINAL RANKING — TIME TO FIRST TOKEN")
    print("=" * 78)

    ranked_ttft = sorted(
        [r for r in all_results if r["avg_first"] is not None],
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
