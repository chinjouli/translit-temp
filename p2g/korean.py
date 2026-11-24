"""IPA -> Hangul: invert epitran's kor-Hang map, syllabify, compose syllable blocks.

`loan=True` applies data/kor-Hang/ (pre rules + substitution table, 외래어 표기법-style,
e.g. f -> ㅍ, ʃ -> 시); stops go by sound: kʰ -> ㅋ, k -> ㄱ. `loan=False` maps each
phone to its feature-nearest Korean phone (faithful mode).

Usage: python korean.py [--faithful] IPA [IPA ...]   (one word per argument)
"""

import csv
import sys
from pathlib import Path

import epitran
from engine import Target, closest
from phones import NONSYL, normalize

MAP = Path(epitran.__file__).parent / "data/map/kor-Hang.csv"
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
CODA = {"k": "ㄱ", "n": "ㄴ", "t": "ㅅ", "l": "ㄹ", "m": "ㅁ", "p": "ㅂ", "ŋ": "ㅇ"}
ASPIRATED_CODA = {"kʰ": "k", "tʰ": "t", "pʰ": "p"}  # loan: 북, 컷 after a short vowel
GLIDE = {"j": "j", "i̯": "j", "ɪ̯": "j", "w": "w", "u̯": "w", "ʊ̯": "w", "ɥ": "w",
         "y̯": "w", "ɰ": "ɰ"}
GLIDE_ALONE = {"j": "i", "w": "u", "ɰ": "ɯ"}
EPENTHESIS = {"t͡ɕ": "i", "t͡ɕʰ": "i"}  # default ㅡ (ɯ); 브리지-style ㅣ after ㅈ/ㅊ
TARGET = Target("kor-Hang")


def _load():
    cons, vows = {}, {}
    with open(MAP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            o, p = r["Orth"], normalize(r["Phon"])
            if o in CHO:
                cons.setdefault(p, o)
            elif o in JUNG:
                vows.setdefault(p, o)  # first wins: we -> ㅞ, not ㅚ
    cons["ɾ"] = "ㄹ"
    return cons, vows


CONS, VOWS = _load()
MONO = tuple(v for v in VOWS if len(v) == 1)


def units(ipa, loan=True):
    """IPA -> [(kind, Korean phone)], kind C or V, unknown phones substituted."""
    raw = []
    for s in TARGET.prepare(ipa, loan):
        if s in GLIDE:
            raw.append(("G", GLIDE[s]))
            continue
        s = s.replace(NONSYL, "")
        if s not in VOWS and s not in CONS:
            s = closest(s, tuple(CONS), MONO)
            if s is None:
                continue  # not IPA: drop
        raw.append(("V" if s in VOWS else "C", s))
    out, i = [], 0
    while i < len(raw):
        kind, p = raw[i]
        if kind == "G":
            nxt = raw[i + 1] if i + 1 < len(raw) else None
            if nxt and nxt[0] == "V" and p + nxt[1] in VOWS:
                out.append(("V", p + nxt[1]))
                i += 2
                continue
            if nxt and nxt[0] == "V" and nxt[1] == GLIDE_ALONE[p]:
                i += 1  # ji, wu, ɰɯ: the vowel already has the glide (ɾʲi -> 리)
                continue
            kind, p = "V", GLIDE_ALONE[p]
        out.append((kind, p))
        i += 1
    return out


def _coda(syl, c, loan):
    """Korean coda for consonant `c` after syllable `syl`, or None if not allowed."""
    if syl[2] is not None:
        return None
    if c in CODA:
        return c
    short = syl[0] is not None and len(syl[1]) == 1
    return ASPIRATED_CODA.get(c) if loan and short else None


def _epenthetic(c):
    return [None, "ɯ", "ŋ"] if c == "ŋ" else [c, EPENTHESIS.get(c, "ɯ"), None]


def syllables(us, loan=True):
    """[(kind, phone)] -> [[onset, nucleus, coda]] with Korean phonotactics."""
    syls, pending = [], []
    for kind, p in us:
        if kind == "C":
            pending.append(p)
            continue
        onset = pending.pop() if pending else None
        if onset == "ŋ":
            pending.append(onset)
            onset = None
        if syls and not pending and onset == "l" and syls[-1][2] is None:
            syls[-1][2] = "l"  # 콜라: intervocalic [l] is ㄹㄹ; a single ㄹ reads [ɾ]
        _close(syls, pending, loan)
        syls.append([onset, p, None])
        pending = []
    _close(syls, pending, loan)
    return syls


def _close(syls, pending, loan):
    """Attach the first pending consonant as a coda if allowed; epenthesize the rest."""
    coda = _coda(syls[-1], pending[0], loan) if syls and pending else None
    if coda:
        syls[-1][2] = coda
        pending.pop(0)
    syls += [_epenthetic(c) for c in pending]


def block(onset, nucleus, coda):
    lead = CHO.index(CONS[onset] if onset else "ㅇ")
    tail = JONG.index(CODA[coda]) if coda else 0
    return chr(0xAC00 + (lead * 21 + JUNG.index(VOWS[nucleus])) * 28 + tail)


def to_hangul(ipa, loan=True):
    """One word of IPA -> Hangul."""
    hangul = "".join(block(*s) for s in syllables(units(ipa, loan), loan))
    return TARGET.finish(hangul, loan)


if __name__ == "__main__":
    args = sys.argv[1:]
    loan = "--faithful" not in args
    for word in (a for a in args if a != "--faithful"):
        print(word, to_hangul(word, loan))
