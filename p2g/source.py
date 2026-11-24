"""Source side: orthography -> IPA in our training conventions.

epitran G2P per word, then the normalization of powsm's data_prep.py (clean at :229,
clean_english at :285), ported rule for rule since data_prep.py cannot be imported
(heavy deps, builds a trie from a relative path on import). English first maps
Flite's conventions to the data's: aj -> aɪ, ej -> eɪ, ow -> oʊ, aw -> aʊ, oj -> ɔɪ,
syllabic ɹ̩ -> ɜ˞, and every l -> l̴ (clean_english restores l where it is not
post-vocalic). One word per call, so the word start is the boundary that
clean_english reads as a preceding space.
"""

import string
import subprocess
import unicodedata
from functools import lru_cache
from pathlib import Path

import epitran
from epitran.rules import Rules
from phones import segments

DATA = Path(__file__).parent / "data"

ENGLISH = {"eng-Latn"}
CUSTOMIZED = {"ʡ": "ʔ", "ᶑ": "ɗ", "g": "ɡ"}  # data_prep.clean
FLITE = {("a", "j"): ["a", "ɪ"], ("e", "j"): ["e", "ɪ"], ("o", "w"): ["o", "ʊ"],
         ("a", "w"): ["a", "ʊ"], ("o", "j"): ["ɔ", "ɪ"], ("ɔ", "j"): ["ɔ", "ɪ"]}
VOWELS = {"i", "ɪ", "e", "ɛ", "æ", "u", "ʊ", "o", "ɔ", "ɑ", "ə", "ɜ˞", "ʌ"}
DEVOICE = {"b": "p", "d": "t", "ɡ": "k"}
STOPS, GLIDES = {"p", "t", "k"}, {"l", "r", "w", "y"}  # Flite ARPAbet
NASALIZE = {v: v + "̃" for v in "iɪeɛæuʊoɔɑ"}


@lru_cache(maxsize=None)
def _epitran(code):
    return epitran.Epitran(code)


def clean(ipa):
    """data_prep.clean: NFD, punctuation removed, a few symbols normalized."""
    ipa = unicodedata.normalize("NFD", ipa)
    ipa = ipa.translate(str.maketrans("", "", string.punctuation))
    return "".join(CUSTOMIZED.get(c, c) for c in ipa)


def flite_to_data(phones):
    """Flite's English IPA conventions -> the data's (before clean_english)."""
    out, i = [], 0
    while i < len(phones):
        pair = tuple(phones[i : i + 2])
        if pair in FLITE:
            out += FLITE[pair]
            i += 2
            continue
        out.append("l̴" if phones[i] == "l" else phones[i])
        i += 1
    return out


def clean_english(tokens):
    """data_prep.clean_english on one word; tokens[0] is the word boundary " "."""
    for i, phone in enumerate(tokens):
        prevp = tokens[i - 1] if i > 0 else None
        nextp = tokens[i + 1] if i < len(tokens) - 1 else None
        if phone in {"p", "t", "k"} and prevp == " " and nextp not in {"s", "ʃ"}:
            tokens[i] = phone + "ʰ"  # 1. word-initial voiceless plosives aspirated
        elif phone in DEVOICE and prevp == " ":  # 2. word-initial b d ɡ devoiced
            if phone == "d" and nextp == "z":
                continue
            elif phone == "d" and nextp == "ʒ":
                tokens[i], tokens[i + 1] = "t", "ʃ"
            else:
                tokens[i] = DEVOICE[phone]
        elif phone == "l̴" and prevp not in VOWELS:
            tokens[i] = "l"  # 3. velarized l only after vowels
        elif phone in {"m", "n", "ŋ"} and prevp in NASALIZE:
            tokens[i - 1] = NASALIZE[prevp]  # 4. vowels nasalized before nasals
    return tokens


def to_ipa(code, word, stress=True):
    """One word in `code`'s orthography -> IPA as the training data writes it.
    English with `stress` also aspirates stressed-syllable stops (flite_ipa)."""
    if code in ENGLISH:
        return normalize(code, flite_ipa(word, stress))
    return normalize(code, _epitran(code).transliterate(word))


def flite_ipa(word, stress=True):
    """English G2P as epitran's Flite (first lex_lookup line), keeping Flite's stress
    digits: with `stress`, p t k right before a stressed vowel (1 or 2), across
    l r w y, and not after s, get ʰ (hotel = hh ow0 t eh1 l -> hoʊtʰɛl). data_prep
    aspirates only word-initial stops, so this is off where the data must match."""
    epi = _epitran("eng-Latn").epi
    text = epi.normalize(word).lower()
    lines = subprocess.run(["lex_lookup", text], capture_output=True,
                           text=True).stdout.splitlines()
    arpa = epi.arpa_text_to_list(lines[0].strip()) if lines else []
    out = []
    for i, a in enumerate(arpa):
        out.append(epi.arpa_map[a.rstrip("012")])
        j = i + 1 + (i + 1 < len(arpa) - 1 and arpa[i + 1] in GLIDES)
        stressed = j < len(arpa) and arpa[j][-1:] in "12"
        if stress and a in STOPS and stressed and (i == 0 or arpa[i - 1] != "s"):
            out[-1] += "ʰ"
    return "".join(out)


@lru_cache(maxsize=None)
def _source_rules(code):
    path = DATA / code / "source.txt"
    return Rules([str(path)]) if path.exists() else None


def normalize(code, ipa):
    """IPA of one word (epitran's, or a dictionary's like WikiPron's, spaces allowed)
    -> the training data's conventions: data_prep.clean, and clean_english for English
    (Flite's aj/ow and syllabic ɹ̩, and WikiPron's ɚ/ɝ, -> aɪ/oʊ and ɜ˞; l -> l̴)."""
    ipa = clean("".join(ipa.split()))
    if rules := _source_rules(code):  # data/<code>/source.txt (deu, tur aspiration)
        ipa = rules.apply(ipa)
    if code not in ENGLISH:
        return ipa
    for rhotic in ("ɹ\u0329", "ɚ", "ɝ"):  # before segments() drops the syllabic mark
        ipa = ipa.replace(rhotic, "ɜ˞")
    phones = flite_to_data(segments(ipa, keep_length=True))
    return "".join(clean_english([" "] + phones)[1:])
