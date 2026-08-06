"""
Grammar-constrained post-processing for GRID-corpus lip-reading output.

GRID sentences always follow a fixed 6-slot template:
    command  color   preposition  letter        digit    adverb
    (4 opts) (4 opts) (4 opts)    (25 opts)     (10 opts) (4 opts)

Instead of routing the raw CTC prediction through an external LLM,
this snaps each predicted word to the nearest VALID word for its slot
using edit distance. It's deterministic, offline, has zero external
dependencies, and the output is still provably your model's own
prediction -- just corrected using the task's known grammar, the same
way real ASR/lip-reading systems use language-model-constrained decoding.
"""

from typing import List, Tuple


# ---- GRID corpus fixed vocabulary, per slot ----
GRID_VOCAB = {
    "command":     ["bin", "lay", "place", "set"],
    "color":       ["blue", "green", "red", "white"],
    "preposition": ["at", "by", "in", "with"],
    "letter":      [c for c in "abcdefghijklmnopqrstuvwxyz" if c != "w"],  # GRID excludes 'w'
    "digit":       ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    "adverb":      ["again", "now", "please", "soon"],
}

SLOT_ORDER = ["command", "color", "preposition", "letter", "digit", "adverb"]


def _levenshtein(a: str, b: str) -> int:
    """Standard edit distance between two strings, no external deps."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                curr_row[j - 1] + 1,      # insertion
                prev_row[j] + 1,          # deletion
                prev_row[j - 1] + cost,   # substitution
            )
        prev_row = curr_row
    return prev_row[-1]


def snap_to_vocab(word: str, slot: str) -> Tuple[str, bool]:
    """
    Snap a single predicted word to the closest valid word for its slot.
    Returns (corrected_word, was_changed).
    """
    vocab = GRID_VOCAB[slot]
    word_lower = word.lower().strip()

    if word_lower in vocab:
        return word_lower, False

    best_word = min(vocab, key=lambda v: _levenshtein(word_lower, v))
    return best_word, True


def grammar_constrained_decode(predicted_words: List[str]) -> dict:
    """
    Main entry point. Pass in the raw 6-word prediction from your model
    (in command, color, preposition, letter, digit, adverb order) and get
    back the corrected sentence plus a per-word change log for your demo
    / report evidence.

    Handles the case where the model doesn't return exactly 6 words too
    (e.g. a dropped or extra token from a bad forward pass) without crashing.
    """
    result = {
        "raw_words": list(predicted_words),
        "corrected_words": [],
        "changes": [],       # list of dicts: {slot, original, corrected}
        "sentence": "",
        "warning": None,
    }

    if len(predicted_words) != len(SLOT_ORDER):
        result["warning"] = (
            f"Expected {len(SLOT_ORDER)} words, got {len(predicted_words)}. "
            f"Slots will be matched positionally up to the shorter length; "
            f"missing slots are left blank."
        )

    for i, slot in enumerate(SLOT_ORDER):
        if i >= len(predicted_words):
            result["corrected_words"].append("")
            continue

        raw = predicted_words[i]
        corrected, changed = snap_to_vocab(raw, slot)
        result["corrected_words"].append(corrected)
        if changed:
            result["changes"].append({
                "slot": slot,
                "original": raw,
                "corrected": corrected,
            })

    result["sentence"] = " ".join(w for w in result["corrected_words"] if w)
    return result


if __name__ == "__main__":
    # Quick sanity check with a deliberately noisy prediction
    example_raw = ["bim", "gren", "at", "z", "sevn", "pleas"]
    out = grammar_constrained_decode(example_raw)
    print("Raw:      ", " ".join(example_raw))
    print("Corrected:", out["sentence"])
    print("Changes:")
    for c in out["changes"]:
        print(f"  [{c['slot']}] '{c['original']}' -> '{c['corrected']}'")
    if out["warning"]:
        print("Warning:", out["warning"])
