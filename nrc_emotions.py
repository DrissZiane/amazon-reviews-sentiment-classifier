#!/usr/bin/env python3
"""Word-list emotion scorer (Assignment Step 5 / part 2).

Implements the "derive it from a word list" take on primary emotion using the NRC
Emotion Lexicon v0.92 (public word->emotion association list). It scores each
review's words against the 8 NRC emotions, sums each emotion's score, and takes the
highest as the review's primary emotion. Requires NO model calls and runs over the
already-saved LLM predictions.

Usage:
    python nrc_emotions.py --input outputs/classify_binary_first100.json \
                           --out outputs/binary_first100_with_nrc.json
    # optionally compare LLM vs NRC emotions:
    python nrc_emotions.py --input ... --out ... --compare
"""

import argparse
import json
import os
import re
from collections import Counter

LEXICON = "lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness",
            "surprise", "trust"]

_WORD_RE = re.compile(r"[a-zA-Z']+")


def load_lexicon(path: str):
    """word -> set of the 8 NRC emotions it is associated with (value==1)."""
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, emotion, flag = parts[0].strip(), parts[1], parts[2]
            if not word or emotion not in EMOTIONS or flag != "1":
                continue
            table.setdefault(word.lower(), set()).add(emotion)
    return table


def nrc_scores(text: str, table: dict):
    """Count-based emotion scores over a review's text + title signal."""
    scores = {e: 0 for e in EMOTIONS}
    for w in _WORD_RE.findall(text.lower()):
        for e in table.get(w, ()):
            scores[e] += 1
    return scores


def primary(scores: dict):
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top, count = ranked[0]
    if count == 0:
        # no lexicon word matched -> honestly "no emotion detected", not a
        # spurious alphabetically-first label (would otherwise all say "anger")
        return None, 0, scores
    return top, count, scores


def augment(path_in, path_out, include_text=True):
    table = load_lexicon(LEXICON)
    rows = json.load(open(path_in, encoding="utf-8"))
    for r in rows:
        # score on body text (+ title contributes as its own signal but we keep it
        # clean: score the text; if empty, fall back to the title)
        text = (r.get("text") or "").strip()
        if not text:
            text = (r.get("title") or "").strip()
        sc = nrc_scores(text, table)
        emo, count, scores = primary(sc)
        r["nrc_emotion"] = emo
        r["nrc_emotion_score"] = count
        r["nrc_scores"] = scores
    os.makedirs(os.path.dirname(path_out) or ".", exist_ok=True)
    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rows)} rows -> {path_out}")
    return rows


def compare(rows):
    """LLM emotion vs NRC-derived emotion agreement."""
    total = len(rows)
    agree = sum(1 for r in rows if (r.get("pred_emotion") or "").lower() == (r.get("nrc_emotion") or "").lower())
    print("\n==== LLM vs NRC primary-emotion comparison ====")
    print(f"agreement: {agree}/{total} = {agree/total:.3f}")
    print("\nconfusion (rows = LLM emotion, cols = NRC emotion):")
    labs = EMOTIONS
    hdr = "        " + "".join(f"{c[:5]:>7s}" for c in labs)
    print(hdr)
    for a in labs:
        row = f"{a:8s} "
        for b in labs:
            row += f"{sum(1 for r in rows if (r.get('pred_emotion') or '').lower()==a and (r.get('nrc_emotion') or '').lower()==b):>7d}"
        print(row)
    # where do they diverge most
    div = Counter()
    for r in rows:
        l = (r.get("pred_emotion") or "").lower()
        n = (r.get("nrc_emotion") or "").lower()
        if l != n:
            div[(l, n)] += 1
    print("\nTop divergences (LLM -> NRC):")
    for (l, n), c in div.most_common(12):
        print(f"  {l:12s} -> {n:12s} : {c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    rows = augment(args.input, args.out)
    if args.compare:
        compare(rows)
