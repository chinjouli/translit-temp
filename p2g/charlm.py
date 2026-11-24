"""Character trigram model of a language's spelling, trained on its WikiPron words.

Scores how plausible a candidate spelling looks in that orthography: chiesa over qiesa
in Italian (qi is rare), e over è in German, no ʒ in French. Add-k smoothing; letters
never seen in the word list get a small probability, so they lose to any real letter.
ngrams() lists the language's frequent 1-3 letter sequences, as spelling candidates.
Built from WikiPron at load (under a second, ~5 s for Russian); not saved.
"""

import csv
import math
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

WIKIPRON = Path(__file__).parents[1] / "wikipron/data/scrape/tsv"
K = 0.1  # add-k smoothing
MIN_WORDS = 1000  # smaller WikiPron files are used only if nothing else exists
MIN_SHARE = 1.5e-4  # rarer letters are loanword-only (French ö, Turkish w)
MAP_SHARE = 5e-5  # floor for letters epitran's map also has (Italian ì ù)


class CharLM:
    def __init__(self, words):
        self.tri, self.bi, self.grams = Counter(), Counter(), Counter()
        for w in words:
            chars = ["^", "^", *w, "$"]
            for a, b, c in zip(chars, chars[1:], chars[2:]):
                self.tri[a, b, c] += 1
                self.bi[a, b] += 1
            for n in (1, 2, 3):
                self.grams.update(w[i : i + n] for i in range(len(w) - n + 1))
        total = sum(n for g, n in self.grams.items() if len(g) == 1)
        self.share = {g: n / total for g, n in self.grams.items() if len(g) == 1}
        self.chars = {g for g, f in self.share.items() if f >= MIN_SHARE}
        self.vocab = len(self.chars) + 2  # + end mark + one unseen letter

    def letters(self, mapped=()):
        """The language's letters: frequent ones, plus `mapped` ones (epitran's map)
        that are not too rare. Drops loanword (French ö) and epitran-internal (õ ʒ)."""
        return self.chars | {c for c in mapped if self.share.get(c, 0) >= MAP_SHARE}

    def ngrams(self, top=600):
        """The most frequent 1-3 letter sequences: candidate spelling units."""
        return [g for g, _ in self.grams.most_common(top)]

    def logprob(self, text, end=True):
        """Total log-probability of `text` as one word (as a word start if not end)."""
        chars = ["^", "^", *unicodedata.normalize("NFC", text.lower())] + ["$"] * end
        return sum(math.log((self.tri[a, b, c] + K) / (self.bi[a, b] + K * self.vocab))
                   for a, b, c in zip(chars, chars[1:], chars[2:]))


def word_list(code):
    """WikiPron TSV for an epitran code (broad over narrow, filtered first; a file
    under MIN_WORDS lines, like Catalan's 175-word broad one, goes last), or None."""
    iso, script = code.split("-")[:2]
    files = sorted(WIKIPRON.glob(f"{iso}_{script.lower()}_*.tsv"),
                   key=lambda f: (_lines(f) < MIN_WORDS, "narrow" in f.name,
                                  "filtered" not in f.name, f.name))
    return files[0] if files else None


def _lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


@lru_cache(maxsize=None)
def model(code):
    """CharLM from the language's WikiPron words, or None if WikiPron lacks it."""
    path = word_list(code)
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        words = {unicodedata.normalize("NFC", r[0]).lower()
                 for r in csv.reader(f, delimiter="\t") if r}
    return CharLM(w for w in words if w.isalpha())
