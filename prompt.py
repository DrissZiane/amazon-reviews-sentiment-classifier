"""Reusable classification prompt for Amazon reviews.

Output is strict JSON so it can be read back programmatically. The prompt never
sees or receives the star rating — the rating is only used afterwards to check
the model. The prompt only ever sees the review's title and text.

Two acceptable sentiment label-sets (chosen by the caller):
    * binary  : POSITIVE | NEGATIVE        (Assignment Step 1 / Step 2)
    * three   : POSITIVE | NEUTRAL | NEGATIVE   (Assignment Step 6+)
"""

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness",
            "surprise", "trust"]

# Examples are deliberately not rating-driven and cover tricky edge cases so the
# model follows the label set and the JSON shape. Titles and texts only.
FEWSHOT = [
    {
        "role": "user",
        "content": (
            "TITLE: Does not work at all\n"
            'TEXT: "Charged it, put it on the card, and the balance never appeared. '
            'Customer service ignored my ticket. Total waste of money."'),
    },
    {
        "role": "assistant",
        "content": (
            '{"sentiment": "NEGATIVE", "emotion": "anger", "confidence": 0.93, '
            '"reason": "Balance never loaded and support ignored the user."}'),
    },
    {
        "role": "user",
        "content": (
            "TITLE: Easy\n"
            'TEXT: "Redeemed in about a minute and the money was in my account right away."'),
    },
    {
        "role": "assistant",
        "content": (
            '{"sentiment": "POSITIVE", "emotion": "joy", "confidence": 0.9, '
            '"reason": "Fast, smooth redemption experience."}'),
    },
]


def make_system_prompt(mode: str = "three") -> str:
    """Build the system prompt.

    mode: 'binary' -> POSITIVE|NEGATIVE
          'three'  -> POSITIVE|NEUTRAL|NEGATIVE
    """
    assert mode in ("binary", "three"), mode
    if mode == "binary":
        labels = "POSITIVE or NEGATIVE"
        label_note = (
            "  - POSITIVE: the writer is clearly satisfied.\n"
            "  - NEGATIVE: the writer is clearly dissatisfied.\n"
            "There is NO neutral option in this mode: choose the closer of the two."
        )
    else:
        labels = "POSITIVE, NEUTRAL or NEGATIVE"
        label_note = (
            "  - POSITIVE: clearly satisfied.\n"
            "  - NEUTRAL: mixed, ambiguous, indifferent, or a balanced \u2014 "
            "neither praise nor complaint.\n"
            "  - NEGATIVE: clearly dissatisfied."
        )

    return f"""You classify Amazon product reviews (the short title line and the body text)
into a sentiment label and a primary emotion. You are given ONLY the review title
and text. You must NEVER see or infer the star rating.

Output exactly ONE JSON object with no extra text, no markdown fences, on a single
line, with these keys:
  "sentiment": {labels},
  "emotion":    one of {EMOTIONS} (the dominant emotion in the text),
  "confidence": a number 0-1 reflecting how sure you are,
  "reason":     a short phrase explaining the call.

Rules:
  - Base your call on the TEXT BODY first; the TITLE second. If the title and body
    conflict, trust the body.
  - A short or terse review is still a real signal: judge its overall tone, not
    its length.
  - Pick the emotion the writer most strongly conveys.
  - Sentiment definition:
{label_note}
Only return the JSON object."""


def build_messages(title: str, text: str, mode: str = "three"):
    """Messages for the chat completion: system + few-shot + the target review."""
    user = f"TITLE: {title}\nTEXT: {text}"
    return [
        {"role": "system", "content": make_system_prompt(mode)},
        *FEWSHOT,
        {"role": "user", "content": user},
    ]


def parse_output(text: str) -> dict:
    """Parses the model's raw text into a dict, tolerating JSON fences/noise."""
    import json
    import re as _re

    t = text.strip()
    # strip markdown fences if present
    fence = _re.search(r"```(?:json)?\s*(.*?)```", t, _re.S)
    if fence:
        t = fence.group(1).strip()
    # if the model wrapped JSON in broader prose, grab the first {...}
    brace = _re.search(r"\{.*\}", t, _re.S)
    if brace:
        t = brace.group(0)
    try:
        return json.loads(t)
    except Exception:
        # last resort: pull a quoted sentiment and emotion out of plain text
        out = {
            "sentiment": None,
            "emotion": None,
            "confidence": None,
            "reason": t[:120],
        }
        for lab in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
            if lab.lower() in t.lower():
                out["sentiment"] = lab
                break
        lo = t.lower()
        for emo in EMOTIONS:
            if emo in lo:
                out["emotion"] = emo
                break
        return out
