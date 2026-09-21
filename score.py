#!/usr/bin/env python3
"""Scoring script: run the Amazon Gift Cards reviews through the LLM classifier
and compare predictions against the "correct answer" derived from the star rating.

The rating is NEVER passed to the model — it is obtained for each review only to
score afterwards.

Usage:
    # Step 2: first 100 rows in file order, binary labels (POSITIVE/NEGATIVE)
    python score.py --mode binary --sample first --n 100 --out outputs/classify_binary_first100.json

    # Step 6: balanced three-class run, ~50 per class, fixed seed (same set every run)
    python score.py --mode three --sample balanced --n 50 --seed 42 --out outputs/classify_balanced_three.json

Each run writes a raw JSON file of per-review predictions (the "working" output)
plus a compact metrics summary printed to the console.
"""

import argparse
import gzip
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

import prompt as P

BASE_URL = os.environ.get("LLM_BASE_URL", "http://dobolyi.com:9001/v1")
API_KEY = os.environ.get("LLM_API_KEY", "6418")
MODEL = os.environ.get("LLM_MODEL", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
DATA = "data/Gift_Cards.jsonl.gz"
WORKERS = 8
MAX_ATTEMPTS = 6

# a prediction that is not a valid label (model failed / nothing returned)
INVALID = "__ERROR__"


def correct_binary(rating: float) -> str:
    return "POSITIVE" if rating >= 4 else "NEGATIVE"


def correct_three(rating: float) -> str:
    if rating >= 4:
        return "POSITIVE"
    if rating == 3:
        return "NEUTRAL"
    return "NEGATIVE"


def load_all_reviews(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sample_first(path: str, n: int):
    return [r for i, r in zip(range(n), load_all_reviews(path))]


def sample_balanced(path: str, n_per_class: int, seed: int):
    """Balanced sample across the three rating classes, fixed-seed so the same set
    returns every run."""
    buckets = {"POSITIVE": [], "NEUTRAL": [], "NEGATIVE": []}
    for r in load_all_reviews(path):
        buckets[correct_three(r["rating"])].append(r)
    rng = random.Random(seed)
    sample = []
    for cls in buckets:
        pool = buckets[cls]
        sample.extend(rng.sample(pool, min(n_per_class, len(pool))))
    rng.shuffle(sample)  # interleave classes in the saved file
    return sample


def classify(client, review, mode: str, max_attempts: int = MAX_ATTEMPTS):
    """Classify one review with retries that VARY the request config so a dead-end
    (silent reasoning that returns nothing) has a fresh chance to succeed.

    The model intermittently returns empty content when its thinking consumes the
    token budget; alternating reasoning off/on, token headroom and temperature
    escapes that dead-end."""
    title = review.get("title", "")
    text = review.get("text", "")
    variants = [
        {"max_tokens": 900, "thinking": False, "temperature": 0.0},
        {"max_tokens": 2600, "thinking": None, "temperature": 0.0},
        {"max_tokens": 1400, "thinking": False, "temperature": 0.6},
        {"max_tokens": 2600, "thinking": None, "temperature": 0.6},
    ]
    last_err = ""
    for i in range(max_attempts):
        v = variants[i % len(variants)]
        try:
            extra = {} if v["thinking"] is None else {"thinking": v["thinking"]}
            resp = client.chat.completions.create(
                model=MODEL,
                messages=P.build_messages(title, text, mode),
                temperature=v["temperature"],
                max_tokens=v["max_tokens"],
                extra_body=extra or None,
            )
            raw = resp.choices[0].message.content
            if raw:
                return P.parse_output(raw), raw
            last_err = "empty content (finish=%s)" % resp.choices[0].finish_reason
        except Exception as e:  # network / server hiccups
            last_err = "%s: %s" % (type(e).__name__, str(e)[:120])
        time.sleep(0.35 + 0.3 * (i % 3))
    return {"sentiment": None, "emotion": None, "confidence": None,
            "reason": "ERROR: %s" % last_err}, None


def run(mode: str, reviews, out_path: str, seed: int):
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    correct = correct_binary if mode == "binary" else correct_three
    label_set = ("POSITIVE", "NEGATIVE") if mode == "binary" else ("POSITIVE", "NEUTRAL", "NEGATIVE")

    results = [None] * len(reviews)
    t0 = time.time()
    done = 0

    def work(review):
        s = time.time()
        parsed, raw = classify(client, review, mode)
        return review, parsed, raw, time.time() - s

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(work, r): i for i, r in enumerate(reviews)}
        for fut in as_completed(futs):
            i = futs[fut]
            review, parsed, raw, _ = fut.result()
            truth = correct(review["rating"])
            results[i] = {
                "seed": seed, "mode": mode,
                "rating": review["rating"],
                "title": review["title"], "text": review["text"],
                "verified_purchase": review["verified_purchase"],
                "helpful_vote": review["helpful_vote"],
                "timestamp": review["timestamp"],
                "truth": truth,              # from rating, never shown to the model
                "pred_sentiment": parsed.get("sentiment"),
                "pred_emotion": parsed.get("emotion"),
                "confidence": parsed.get("confidence"),
                "reason": parsed.get("reason"),
                "raw_model_output": raw,
                "correct": parsed.get("sentiment") == truth,
            }
            done += 1
            if done % 25 == 0 or done == len(reviews):
                print(f"  classified {done}/{len(reviews)}  ({round(time.time()-t0,1)}s)",
                      flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return summarize(results, mode, label_set)


def summarize(results, mode, label_set):
    n = len(results)
    valid = [r for r in results if r["pred_sentiment"] in label_set]
    correct = [r for r in valid if r["correct"]]
    acc = len(correct) / n
    nerr = n - len(valid)
    print("\n========== SUMMARY (mode=%s, n=%d) ==========" % (mode, n))
    print(f"Overall accuracy (model agrees with rating-based label): {acc:.3f}  ({len(correct)}/{n})")
    print(f"  of which {nerr} row(s) had an invalid/empty model output (counted as wrong):")

    truth_counts = {c: 0 for c in label_set}
    for r in results:
        truth_counts[r["truth"]] = truth_counts.get(r["truth"], 0) + 1
    print("\nTruth (rating-derived) class distribution:")
    for c in label_set:
        print(f"  {c:8s}: {truth_counts.get(c,0)}")

    cols = label_set + (INVALID,)
    print("\nConfusion matrix (rows=truth from rating, cols=prediction):")
    header = "        " + "".join(f"{c[:8]:>10s}" for c in cols)
    print(header)
    for t in label_set:
        row = f"{t:8s} "
        for p in cols:
            row += f"{sum(1 for r in results if r['truth']==t and (r['pred_sentiment'] if r['pred_sentiment'] in label_set else INVALID)==p):>10d}"
        print(row)

    print("\nPer-class recall (true->predicted correctly):")
    for c in label_set:
        tot = truth_counts.get(c, 0)
        hit = sum(1 for r in results if r["truth"] == c and r["pred_sentiment"] == c)
        print(f"  {c:8s}: {hit}/{tot} = {hit/tot:.3f}" if tot else f"  {c:8s}: n/a")

    print("\nPer-class precision (predicted->is actually that):")
    for p in label_set:
        pred_tot = sum(1 for r in results if r["pred_sentiment"] == p)
        hit = sum(1 for r in results if r["pred_sentiment"] == p and r["truth"] == p)
        print(f"  {p:8s}: {hit}/{pred_tot} = {hit/pred_tot:.3f}" if pred_tot else f"  {p:8s}: n/a")

    wrong = [r for r in results if not r["correct"]]
    print(f"\nMismatches ({len(wrong)}):")
    for r in wrong:
        pred = r["pred_sentiment"] if r["pred_sentiment"] in label_set else "<none/invalid>"
        print(f"  ★{int(r['rating'])}  pred={pred:8s} truth={r['truth']:8s} | {r['title'][:50]!r} {r['text'][:70]!r}")
    return {"accuracy": acc, "n": n, "mode": mode, "correct": len(correct)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["binary", "three"], default="binary")
    ap.add_argument("--sample", choices=["first", "balanced"], default="first")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-per-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.sample == "first":
        reviews = sample_first(DATA, args.n)
        seed_for_out = 0
    else:
        reviews = sample_balanced(DATA, args.n_per_class, args.seed)
        seed_for_out = args.seed

    if not args.out:
        fname = f"outputs/classify_{args.mode}_{args.sample}"
        if args.sample == "first":
            fname += f"_{args.n}"
        else:
            fname += f"_n{args.n_per_class}_seed{args.seed}"
        args.out = fname + ".json"

    print(f"Mode={args.mode} sample={args.sample} -> {args.out}")
    run(args.mode, reviews, args.out, seed_for_out)
