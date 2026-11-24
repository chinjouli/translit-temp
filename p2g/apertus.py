"""Apertus tokenizer piece ranks: frequency proxy for spellings (lower = commoner)."""

import os
from functools import lru_cache
from pathlib import Path

TOKENIZER = Path(os.environ["APERTUS_TOKENIZER"])


@lru_cache(maxsize=None)
def _ranks():
    from tokenizers import Tokenizer
    tok, out = Tokenizer.from_file(str(TOKENIZER)), {}
    for i in range(tok.get_vocab_size()):
        out.setdefault(tok.decode([i]).strip(), i)
    return out


def rank(text):
    """Token id of `text` as a single piece (case-folded fallback), else infinity."""
    ranks = _ranks()
    return ranks.get(text, ranks.get(text.lower(), float("inf")))
