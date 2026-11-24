"""Round-trip scoring: spell -> epitran forward -> phone error rate vs the target."""

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import epitran
from phones import read_ipa, segments, sub_cost

ENGLISH = os.environ.get("P2G_ENGLISH", "flite")  # English reader: flite or g2p_en
SLOW = {"eng-Latn"} if ENGLISH == "flite" else set()  # Flite: a process per word
_CACHE = {}
_THREADS = ThreadPoolExecutor(16)


@lru_cache(maxsize=None)
def _epitran(code):
    return epitran.Epitran(code)


@lru_cache(maxsize=None)
def _g2p_en():
    from g2p_en import G2p
    return G2p()


def _read(code, text):
    epi = _epitran(code)
    if code == "eng-Latn" and ENGLISH == "g2p_en":  # CMUdict + neural guess, ARPAbet
        arpa = " ".join("ax" if p == "AH0" else p.lower()  # schwa, as Flite writes it
                        for p in _g2p_en()(text) if p.strip())
        ipa = epi.epi.arpa_to_ipa(f"({arpa})")
    else:
        ipa = epi.transliterate(text)
    return tuple(segments(read_ipa(ipa), True))


def forward_many(code, texts):
    """Spellings -> phones, read back by epitran; cached, Flite calls in threads."""
    todo = list({t for t in texts if (code, t) not in _CACHE})
    run = _THREADS.map if code in SLOW and len(todo) > 1 else map
    _epitran(code)
    for text, phones in zip(todo, run(lambda t: _read(code, t), todo)):
        _CACHE[code, text] = phones
    return [_CACHE[code, t] for t in texts]


def forward(code, text):
    """Spelling -> phones, read back by epitran."""
    return forward_many(code, [text])[0]


def distance(ref, hyp, weighted=False):
    """Levenshtein distance between phone sequences; substitutions cost their feature
    distance (phones.sub_cost) if weighted, else 1."""
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            sub = sub_cost(r, h) if weighted else r != h
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + sub)
    return row[-1]


def per(ref, hyp):
    """Phone error rate of `hyp` against `ref`."""
    return distance(ref, hyp) / max(len(ref), 1)
