"""IPA -> target script; one entry point for every target.

TARGET is an epitran-style code or an alias: pinyin, bopomofo, kor, jpn (katakana),
eng (romanization, eng-Roman; eng-Latn is the Flite-read respelling), or the ISO 639-3
code of any built language (data/<code>/align_wikipron.tsv: fra -> fra-Latn; 66
alphabets and abjads).
The module: Hangul -> korean.py; data/<code>/roman.csv -> roman.py;
data/<code>/syllables.txt -> syllabary.py; else alphabet.py.

With --orth SRC the inputs are words in SRC's orthography (an epitran code): they go
through epitran G2P and our training-data normalization first (source.py). For
English, stops starting a stressed syllable are aspirated too, from Flite's stress
marks (hotel -> hoʊtʰɛl̴); --no-stress keeps the data's word-initial-only rule.

Usage: python translit.py TARGET IPA [IPA ...]   (one word per argument)
       python translit.py TARGET --orth SRC WORD [WORD ...]
       e.g. python translit.py bopomofo 'pʰaʁi'
            python translit.py kor --orth eng-Latn computer hotel
"""

import argparse
from functools import lru_cache

import alphabet
import roman
import source
import syllabary
from engine import DATA

ALIASES = {"kor": "kor-Hang", "pinyin": "cmn-Latn2", "bopomofo": "cmn-Bopo2",
           "eng": "eng-Roman", "jpn": "jpn-Kana"}


def resolve(name):
    """Alias or ISO 639-3 code of a built language -> epitran code."""
    if name in ALIASES:
        return ALIASES[name]
    built = [p.parent.name for p in DATA.glob(f"{name}-*/align_wikipron.tsv")]
    return built[0] if len(built) == 1 else name


def reader(name):
    """epitran code that reads `name`'s spelling (eng -> eng-Latn, Flite)."""
    code = resolve(name)
    return "eng-Latn" if code == "eng-Roman" else code


@lru_cache(maxsize=None)
def converter(target):
    """Callable IPA -> spelling for `target` (code or alias)."""
    code = resolve(target)
    if code == "kor-Hang":
        from korean import to_hangul
        return to_hangul
    if (DATA / code / "roman.csv").exists():
        return roman.converter(code)
    if (DATA / code / "syllables.txt").exists():
        return syllabary.converter(code)
    return alphabet.converter(code)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="IPA (or --orth words) -> target script")
    ap.add_argument("target", help="epitran code, ISO 639-3 code of a built "
                    "language, or alias: " + " ".join(ALIASES))
    ap.add_argument("words", nargs="+")
    ap.add_argument("--orth", metavar="SRC", help="inputs are SRC orthography")
    ap.add_argument("--no-stress", action="store_true", help="English --orth only: "
                    "aspirate word-initial stops only, as the training data does")
    args = ap.parse_args()
    conv = converter(args.target)
    for word in args.words:
        ipa = (source.to_ipa(reader(args.orth), word, not args.no_stress)
               if args.orth else word)
        print(word, ipa, conv(ipa), sep="\t") if args.orth else print(word, conv(ipa))
