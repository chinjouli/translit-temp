"""IPA -> conventional romanization, the English target (`eng`): one phone, one
spelling, from data/<code>/roman.csv (text-pilot's ROMAN, extended); no read-back.

Continental vowels (a e i o u) with English consonant digraphs (sh zh ch j th ng kh).
A phone not in the table is tried without its diacritics, length and modifiers
(kʰ -> k, l̴ -> l), then through data/default/fallback.csv, then as the feature-nearest
listed phone, so nothing is dropped. An r-coloured vowel is the vowel + r, a nasal
vowel the vowel + n unless a nasal consonant follows (bonzhur, spin). Split-off
palatalization is y before a back vowel (мясо -> myaso) and nothing elsewhere
(мартин -> martin); labialization likewise w. Optional pre.txt/subst.csv/post.txt.
"""

import csv
import unicodedata
from functools import lru_cache

from engine import DATA, FALLBACK, Target, closest
from phones import MODIFIERS, is_vowel

FRONT = set("ieɪɛ")
NASAL_MARK, NASALS = "\u0303", set("mnŋɲɱɳ")
GLIDE = {"ʲ": "y", "ʷ": "w"}


def _nfd(text):
    return unicodedata.normalize("NFD", text)


class Roman:
    def __init__(self, code="eng-Roman"):
        with open(DATA / code / "roman.csv", encoding="utf-8") as f:
            self.table = {_nfd(r["Phon"]): r["Letters"] for r in csv.DictReader(f)}
        self.target, self.code = Target(code), "eng-Latn"  # code: G2P reader
        listed = [p for p in self.table if len(p) == 1]
        self.vowels = tuple(p for p in listed if is_vowel(p))
        self.consonants = tuple(p for p in listed if not is_vowel(p))

    def phones(self, ipa):
        return self.target.prepare(ipa, secondary={"ʲ": "ʲ", "ʷ": "ʷ"})

    @lru_cache(maxsize=None)
    def letters(self, p):
        """Spelling of one phone."""
        if p in self.table:
            return self.table[p]
        if "˞" in p:
            vowel = self.letters(p.replace("˞", ""))
            return vowel if vowel.endswith("r") else vowel + "r"
        base = "".join(c for c in p if not unicodedata.combining(c)
                       and c not in MODIFIERS) or p
        if base in self.table:
            return self.table[base]
        if base.isascii():
            return base
        if base in FALLBACK:
            hits = [f for f in FALLBACK[base] if f in self.table]
            return self.table[hits[0]] if hits else ""
        near = closest(base, self.consonants, self.vowels)
        return self.table[near] if near else ""

    def __call__(self, ipa):
        phones, out = self.phones(ipa), []
        for i, p in enumerate(phones):
            if p in GLIDE:
                nxt = phones[i + 1] if i + 1 < len(phones) else ""
                vowel = nxt and is_vowel(nxt) and nxt[0] not in FRONT
                out.append(GLIDE[p] if vowel else "")
            else:
                nxt = phones[i + 1] if i + 1 < len(phones) else ""
                nasal = NASAL_MARK in p and nxt[:1] not in NASALS  # bɔ̃ʒuʀ: bonzhur
                out.append(self.letters(p) + "n" * nasal)
        return self.target.finish("".join(out))


@lru_cache(maxsize=None)
def converter(code="eng-Roman"):
    return Roman(code)
