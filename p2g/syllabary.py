"""IPA -> a script written in a closed set of syllables (pinyin, Bopomofo, kana, ...).

Dynamic programming cuts the input into syllables, snapping each piece to the nearest
listed syllable by weighted edit distance: substitution cost grows with phonological
feature distance (high vowel <-> its glide is nearly free), and adding a vowel is
cheaper than dropping a consonant, so clusters gain syllables (strike -> si te lai ke).

Everything language-specific is data in data/<code>/:
  syllables.txt   one spelling per line; earlier lines win on equal IPA
  table.txt       epitran-syntax rules on each syllable's IPA (zi = t͡sz̩ -> t͡sɨ)
  syllabary.toml  separator, max_span, keep_length, costs (insert, delete,
                  delete_w, glide, unglide, glide_onset),
                  [epenthetic]
  pre.txt, subst.csv, post.txt as for every target
Syllable IPA is epitran(<code>) of the spelling. Input phones no syllable uses take
their first fallback the target has (data/default/fallback.csv), so h -> x in pinyin.
"""

import tomllib
import unicodedata
from functools import lru_cache

import epitran
from engine import DATA, FALLBACK, Target, _rules
from phones import DST, FT, TIE, is_vowel, segments

GLIDES = {"i̯": "j", "ɪ̯": "j", "u̯": "w", "ʊ̯": "w", "y̯": "ɥ"}
HIGH = {"i": "j", "ɪ": "j", "u": "w", "ʊ": "w", "y": "ɥ"}  # next to a vowel: glide


def glides(phones, onglide=False, offglides=None):
    """Non-syllabic vowels -> glides; a high vowel after a vowel (ai -> aj) too, and
    before one if `onglide` (syllable tables: ㄍㄨㄛ kʊɔ -> kwɔ). With `offglides`
    (vowel, glide) pairs, only diphthongs the target has: boy ɔɪ stays ɔ + i."""
    out = [GLIDES.get(p, p) for p in phones]
    vowel = [is_vowel(p) for p in out]
    for i, p in enumerate(out):
        after = i > 0 and vowel[i - 1]
        if after and offglides is not None and p in HIGH:
            after = (out[i - 1], HIGH[p]) in offglides
        before = onglide and i + 1 < len(out) and vowel[i + 1]
        if p in HIGH and (after or before):
            out[i], vowel[i] = HIGH[p], False
    return tuple(out)


def _redundant(src, k):
    """Onglide w a rounded vowel can carry (kwʊn ~ ㄍㄨㄣ kʊn; not the ow/aw of town)."""
    return src[k] == "w" and not (k > 0 and is_vowel(src[k - 1]))


def _absorbed(phones, k):
    """A glide next to the vowel it is part of: j before or after i/ɪ, w before u
    (Russian bʲi, Japanese ɾʲi -> ㄅㄧ, ㄌㄧ, not ㄅㄧㄝ; ный nij -> ㄋㄧ). Not w before ʊ:
    the map's ㄨㄣ is wʊn."""
    nxt = phones[k + 1] if k + 1 < len(phones) else None
    prev = phones[k - 1] if k else None
    return ((phones[k], nxt) in {("j", "i"), ("j", "ɪ"), ("w", "u")}
            or (phones[k] == "j" and prev in ("i", "ɪ")))


@lru_cache(maxsize=None)
def _feature_cost(a, b):
    if FT.seg_known(a) and FT.seg_known(b):
        return min(1.0, DST.weighted_feature_edit_distance(a, b) / 4)
    return 1.0


class Syllabary:
    def __init__(self, code):
        folder = DATA / code
        with open(folder / "syllabary.toml", "rb") as f:
            cfg = tomllib.load(f)
        self.separator = cfg.get("separator", "")
        self.max_span = cfg.get("max_span", 6)
        self.keep_length = cfg.get("keep_length", False)  # kana: ー for long vowels
        cost = cfg.get("cost", {})
        self.insert, self.delete = cost.get("insert", 0.5), cost.get("delete", 1.0)
        self.delete_w = cost.get("delete_w", 0.2)  # rounding a vowel can carry: ㄍㄨㄣ
        self.syllable = cost.get("syllable", 0.0)  # per syllable: fewer syllables win
        self.glide = cost.get("glide", 0.1)  # input vowel -> syllable glide (medial)
        self.unglide = cost.get("unglide", 0.3)  # input glide -> syllable vowel
        self.glide_onset = cost.get("glide_onset", 0.01)  # ties: ㄌㄞ ㄛ, not ㄌㄚ ㄧㄛ
        self.epenthetic = cfg.get("epenthetic", {})
        self.target, table = Target(code), _rules(folder / "table.txt")
        epi = epitran.Epitran(code)
        self.syllables = {}  # syllable phones -> spelling
        for spelling in (folder / "syllables.txt").read_text(encoding="utf-8").split():
            ipa = epi.transliterate(spelling)
            ipa = unicodedata.normalize("NFD", ipa)  # rules are NFD (ç = c + cedilla)
            ipa = table.apply(ipa) if table else ipa
            self.syllables.setdefault(glides(segments(ipa), onglide=True), spelling)
        self.known = {p for k in self.syllables for p in k}
        self.offglides = {(a, b) for k in self.syllables for a, b in zip(k, k[1:])
                          if is_vowel(a) and b in HIGH.values()}
        self._cache = {}

    def sub_cost(self, a, b):
        """Input phone `a` -> syllable phone `b`. A glide becoming a vowel costs more
        than the reverse: it breaks a diphthong (rain: ㄖㄨㄟ ㄣ, not ㄖㄨ ㄝ ㄧㄣ)."""
        if a == b:
            return 0.0
        if HIGH.get(a) == b:
            return self.glide
        if HIGH.get(b) == a:
            return self.unglide
        return _feature_cost(a, b)

    def distance(self, src, tgt):
        """Weighted edit distance from input phones to syllable phones."""
        ins = [self.epenthetic.get(t, self.insert) for t in tgt]
        row = [0.0]
        for c in ins:
            row.append(row[-1] + c)
        for k, s in enumerate(src):
            delete = self.delete_w if _redundant(src, k) else self.delete
            prev, row[0] = row[0], row[0] + delete
            for j, t in enumerate(tgt, 1):
                sub = prev + self.sub_cost(s, t)
                cost = min(row[j] + delete, row[j - 1] + ins[j - 1], sub)
                prev, row[j] = row[j], cost
        return row[-1]

    def nearest(self, src):
        """(cost, spelling) of the closest syllable to input phones `src`."""
        if src not in self._cache:
            scored = ((self.distance(src, k) + self._onset(k), len(s), s)
                      for k, s in self.syllables.items())  # ties: shorter (ㄣ, not ㄋㄜ)
            cost, _, spelling = min(scored)
            self._cache[src] = cost, spelling
        return self._cache[src]

    def _onset(self, key):
        """Glide-initial syllables lose ties, so a diphthong keeps its offglide."""
        return self.glide_onset if key and key[0] in HIGH.values() else 0.0

    def _known(self, p):
        """`p` if a syllable uses it, else its first fallback that one does (None for an
        empty fallback list: drop it); a phone panphon cannot read either drops its
        diacritics (Korean tense k͈ -> k)."""
        if p in self.known:
            return p
        if FALLBACK.get(p) == []:
            return None
        fallback = next((f for f in FALLBACK.get(p, []) if f in self.known), None)
        if fallback:
            return fallback
        base = "".join(c for c in p if not unicodedata.combining(c) or c in TIE)
        if base != p and not FT.seg_known(p):
            return self._known(base)
        return p

    def __call__(self, ipa, loan=True):
        """One word of IPA -> syllables joined by the separator.

        `loan=False` skips pre.txt/subst.csv/post.txt (the loanword rules), e.g. to
        check the syllable table itself.
        """
        prepared = self.target.prepare(ipa, loan, keep_length=self.keep_length)
        phones = [self._known(p) for p in prepared]
        phones = [p for p in phones if p is not None]  # empty fallback: dropped (ʕ)
        phones = glides(tuple(phones), offglides=self.offglides)
        phones = tuple(p for k, p in enumerate(phones) if not _absorbed(phones, k))
        best = [(0.0, [])] + [(float("inf"), [])] * len(phones)
        for j in range(1, len(phones) + 1):
            for i in range(max(0, j - self.max_span), j):
                cost, spelling = self.nearest(phones[i:j])
                cost += self.syllable
                if best[i][0] + cost < best[j][0]:
                    best[j] = (best[i][0] + cost, best[i][1] + [spelling])
        return self.target.finish(self.separator.join(best[-1][1]), loan)


@lru_cache(maxsize=None)
def converter(code):
    return Syllabary(code)
