"""
GRAMMAR-CONSTRAINED DECODING for GRID-style predictions
=========================================================
GRID sentences always follow a fixed 6-word template:
    command  color  preposition  letter  digit  adverb
e.g. "set    blue   at           f       two    now"

Each slot only ever contains one of a small, fixed set of words.
Your CTC decoder doesn't know this structure -- it just picks the
single most likely word at each timestep, which can produce a word
that's technically valid vocabulary but wrong for that slot's role
(e.g. predicting "please" in the color slot).

This snaps each predicted word to the closest valid word FOR ITS
SLOT, using edit distance. This is a legitimate, standard technique
(grammar/language-model-constrained decoding) -- it's still your
model's own output, just corrected using known task structure, not
an external LLM guessing a plausible sentence.

Fully offline, deterministic, no dependencies beyond the stdlib.
"""

# Standard GRID corpus grammar (Cooke et al. 2006)
COMMANDS     = ['bin', 'lay', 'place', 'set']
COLORS       = ['blue', 'green', 'red', 'white']
PREPOSITIONS = ['at', 'by', 'in', 'with']
LETTERS      = [c for c in 'abcdefghijklmnopqrstuvyz']  # GRID excludes 'w'
DIGITS       = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
ADVERBS      = ['again', 'now', 'please', 'soon']

GRID_SLOTS = [COMMANDS, COLORS, PREPOSITIONS, LETTERS, DIGITS, ADVERBS]


def _edit_distance(a, b):
    """Standard Levenshtein distance, stdlib only."""
    if a == b:
        return 0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + 1,          # deletion
                dp[j - 1] + 1,      # insertion
                prev + (a[i-1] != b[j-1])  # substitution
            )
            prev = cur
    return dp[n]


def _closest(word, candidates):
    """Nearest valid word in this slot's vocabulary, by edit distance."""
    return min(candidates, key=lambda c: _edit_distance(word, c))


def grammar_constrain(hyp_words):
    """
    hyp_words: list of predicted words from CTC decode, e.g.
               ['place', 'white', 'in', 'z', 'six', 'please']
    returns:   same-length list, each word snapped to the nearest
               valid word for its slot. If hyp_words isn't exactly
               6 words (CTC sometimes over/under-predicts), it's
               returned unchanged -- safer than guessing alignment.
    """
    if len(hyp_words) != 6:
        return hyp_words  # can't safely align to slots, leave as-is
    return [_closest(w, slot) for w, slot in zip(hyp_words, GRID_SLOTS)]


# ------------------------------------------------------------
# Quick before/after check, standalone
# ------------------------------------------------------------
if __name__ == '__main__':
    examples = [
        ['place', 'white', 'in', 'z', 'six', 'please'],   # 'z' isn't a GRID letter -> nearest match
        ['set', 'blue', 'with', 'k', 'nine', 'soon'],      # already fully valid
        ['lay', 'red', 'by', 'i', 'four', 'again'],
    ]
    for hyp in examples:
        fixed = grammar_constrain(hyp)
        print("before:", ' '.join(hyp))
        print("after: ", ' '.join(fixed))
        print()
