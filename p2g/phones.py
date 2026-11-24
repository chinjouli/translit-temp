"""IPA segmentation and nearest-phone substitution, shared by all target scripts."""

import unicodedata
from functools import lru_cache

import panphon
import panphon.distance

FT = panphon.FeatureTable()
DST = panphon.distance.Distance()
TIE = "͜͡"
NONSYL, SYLLABIC = "̯", "̩̍"
MODIFIERS = set("ʰʲʷˠˤⁿˡ˞")  # modifier letters that attach to the previous base
LENGTH = set("ːˑ")
DROP = set("ˈˌ.|/-_ ˥˦˧˨˩") | set("⁰¹²³⁴⁵⁶⁷⁸⁹⁻")  # stress, tone, separators


def normalize(ipa, keep_length=False):
    """NFD, drop stress/tone/separators, syllabic marks and (unless kept) length."""
    ipa = unicodedata.normalize("NFD", ipa)
    drop = DROP if keep_length else DROP | LENGTH
    return "".join(c for c in ipa if c not in drop and c not in SYLLABIC)


def read_ipa(ipa):
    """epitran output -> our conventions: Flite's syllabic r (butter = bʌtɹ̩) is əɹ;
    epitran French è before a consonant (père = pə̀ʀ) is ɛ."""
    ipa = unicodedata.normalize("NFD", ipa)
    return ipa.replace("ɹ\u0329", "əɹ").replace("ə\u0300", "ɛ")


def segments(ipa, keep_length=False):
    """Split IPA into base + diacritic segments; keeps marks panphon drops."""
    segs, chars, i = [], normalize(ipa, keep_length), 0
    while i < len(chars):
        seg = chars[i]
        i += 1
        while i < len(chars):
            c = chars[i]
            if c in TIE and i + 1 < len(chars):
                seg += chars[i : i + 2]
                i += 2
            elif unicodedata.combining(c) or c in MODIFIERS or c in LENGTH:
                seg += c
                i += 1
            else:
                break
        segs.append(seg)
    return segs


def is_vowel(seg):
    if NONSYL in seg:
        return False
    fts = FT.fts(seg) if FT.seg_known(seg) else None
    return bool(fts) and fts["syl"] == 1


@lru_cache(maxsize=None)
def nearest(seg, inventory):
    """Closest phone of `inventory` (a tuple) by weighted feature edit distance."""
    return min(inventory, key=lambda p: DST.weighted_feature_edit_distance(seg, p))


@lru_cache(maxsize=None)
def sub_cost(a, b):
    """Substitution cost in [0, 1]: twice the share of differing features (t/d 0.08,
    ə/ø 0.25), 1 if either phone is unknown, 0.1 if only panphon's features agree
    (trill r vs tap ɾ)."""
    if a == b:
        return 0.0
    fa, fb = FT.word_fts(a), FT.word_fts(b)
    if len(fa) != 1 or len(fb) != 1:
        return 1.0
    diff = sum(x != y for x, y in zip(fa[0].numeric(), fb[0].numeric()))
    return min(1.0, 2 * diff / len(fa[0].numeric())) if diff else 0.1
