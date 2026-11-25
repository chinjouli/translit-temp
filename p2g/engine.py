"""Shared IPA -> script steps: pre rules, substitution, post rules.

Per target language, data lives in data/<code>/ (every file optional):
  pre.txt    epitran-syntax rules applied to the IPA before substitution
  subst.csv  Phon,Target: source phone -> space-separated target phones ("" deletes)
  post.txt   epitran-syntax rules applied to the output script
"""

import csv
import unicodedata
from pathlib import Path

from epitran.rules import Rules
from phones import FT, LENGTH, is_vowel, nearest, segments

DATA = Path(__file__).parent / "data"


def _nfd(text):
    """NFD, as segments() produces: ç in a CSV must match c + combining cedilla."""
    return unicodedata.normalize("NFD", text)


with open(DATA / "default/fallback.csv", encoding="utf-8") as _f:  # shared, ordered
    FALLBACK = {_nfd(r["Phon"]): _nfd(r["Fallbacks"]).split()
                for r in csv.DictReader(_f)}
SECONDARY = {"ʲ": "j", "ʷ": "w"}  # kʲ -> k + j, kʷ -> k + w
FRONT = set("ieɪɛ")  # a split-off j is dropped before these (and before j)
UNWRITTEN = str.maketrans("", "", "ˠˤ")  # secondary marks no target spells


class Target:
    """Language-specific data for one output script."""

    def __init__(self, code):
        folder = DATA / code
        self.pre, self.post = (_rules(folder / f) for f in ("pre.txt", "post.txt"))
        self.subst = {}
        if (folder / "subst.csv").exists():
            with open(folder / "subst.csv", encoding="utf-8") as f:
                self.subst = {_nfd(r["Phon"]): _nfd(r["Target"]).split()
                              for r in csv.DictReader(f)}

    def prepare(self, ipa, loan=True, keep_length=False, keep=(), secondary=None):
        """IPA -> phone list: pre rules and substitution table (loan mode only).

        kʲ is split into k + j unless it is in `keep` (e.g. the target map has it),
        and the j dropped before a front vowel or j, where it is implied;
        `secondary` overrides what ʲ/ʷ become (Russian: kʲ -> k + ʲ, spelled ь).
        Without keep_length, length goes before the pre rules so it cannot hide a
        vowel context (aːr is a + r).
        """
        ipa = "".join(ipa.split())  # spaced phones (p a ɾ i): rules must see neighbours
        ipa = ipa.translate(UNWRITTEN)  # velarized ɾˠ, pharyngealized tˤ: plain
        if not keep_length:
            ipa = "".join(c for c in _nfd(ipa) if c not in LENGTH)
        if loan and self.pre:
            ipa = self.pre.apply(ipa)
        out, segs = [], segments(ipa, keep_length)
        for i, s in enumerate(segs):
            split = len(s) > 1 and s[-1] in SECONDARY and s not in keep
            extra = [(secondary or SECONDARY)[s[-1]]] if split else []
            s = s[:-1] if split else s
            nxt = segs[i + 1] if i + 1 < len(segs) else ""
            if extra == ["j"] and (nxt[:1] in FRONT or nxt[:1] == "j"):
                extra = []  # implied: мартин = martʲin -> martin, not martjin
            out += (self.subst.get(s, [s]) if loan else [s]) + extra
        return out

    def finish(self, text, loan=True):
        """post.txt on the output; rules are NFD, so the text is too (ガ = カ + ゙)."""
        if not (loan and self.post):
            return text
        return unicodedata.normalize("NFC", self.post.apply(_nfd(text)))


def _rules(path):
    return Rules([path]) if path.exists() else None


def closest(seg, consonants, vowels):
    """Feature-nearest phone of the matching class, or None if `seg` is not IPA."""
    seg = seg if FT.seg_known(seg) else seg[0]
    if not FT.seg_known(seg):
        return None
    pool = vowels if is_vowel(seg) else consonants
    return nearest(seg, pool) if pool else None  # abjads may map no vowels
