"""IPA -> any alphabetic script with an epitran map: beam search over spellings, each
scored by how the target's epitran reads it back.

Spelling alternatives per pronunciation come from epitran's map (or data/<code>/base.csv
where epitran has none, e.g. English via Flite), minus rows with non-letters (<dZ>) or
letters the language does not write (charlm.letters: French õ ʒ ö), plus the
language's frequent 1-3 letter sequences (charlm.ngrams) under whatever epitran reads
them as: between a stable vowel and between a sonorant (French on = ɔ̃ before a
consonant), and, for the word's end only, alone (German final e = ə; Russian лог alone
is lok). data/<code>/units.txt, when present, adds common orthographic syllables read
whole (English: market, morning). A vowel also tries its nearest vowels' spellings
(French closed-syllable ə: eu). data/<code>/align_wikipron.tsv (scripts/make_align.py)
adds the spellings WikiPron words use, with P(letters | phones, context) as a cost.
The search extends prefixes phone by phone and keeps the BEAM best, each read back with
a vowel letter appended so it reads as mid-word. Since whole prefixes are read, context
decides (French ente = ɑ̃t, ent = ɑ̃). At the end, letters that are silent after a
consonant are tried (French bette = bɛt). Cost: feature-weighted phone edit distance
(t/d cheap, t/a dear), plus a cost per letter and per nat of charlm surprisal, so
short, natural spellings win near-ties (chiesa over qiesa).
A phone absent from the table tries data/default/fallback.csv in order (h -> x χ ħ),
then the feature-nearest phone. Brahmic abugidas (Devanagari, Bengali, …) are spelled
here too, epitran's schwa deletion deciding where a consonant needs no vowel sign
(कमल = kəməl), with spellings kept well formed (_fits); other abugidas and syllabaries
need their own module.
Optional data/<code>/ (pre.txt, subst.csv, post.txt) works as for every target.

Usage: python alphabet.py CODE IPA [IPA ...]   e.g. python alphabet.py spa-Latn peɾo
"""

import csv
import math
import sys
import unicodedata
from collections import Counter, defaultdict
from functools import cached_property, lru_cache
from pathlib import Path

import epitran
from charlm import model
from engine import DATA, FALLBACK, SECONDARY, Target, closest
from phones import LENGTH, is_vowel, segments, sub_cost
from roundtrip import distance, forward, forward_many

MAP_DIR = Path(epitran.__file__).parent / "data/map"
MAX_ALTS = 4  # spellings tried per chunk
BEAM = 3  # prefixes kept per phone position (5: +0.01 back, 1.6x slower; 1: -0.04-0.08)
MAX_LETTERS = 3  # letters per phone (+ 2) a spelling may use
WINDOW = 8  # letters re-read per step once a spelling passes 2 x WINDOW (long input)
LETTER = 0.03  # cost per letter: shorter spellings win near-ties (cf. syllabary.toml)
SURPRISAL = 0.02  # cost per nat of charlm surprisal: natural letters win near-ties
NEAR = 2  # nearest vowels whose spellings a vowel also tries (French ə: eu)
ALIGN = 0.02  # cost per nat of -log P(letters | phones, context) from ALIGN_FILE
UNSEEN = 6.0  # -log P for a spelling the alignment never saw (P ~ 0.25%)
MIN_ALIGN = 3  # alignment pairs seen fewer times are noise
ALIGN_FILE = "align_wikipron.tsv"  # in data/<code>/ (scripts/make_align.py)
NO_ALIGN = {"eng-Latn"}  # too irregular: the prior made respellings worse (bosure)
MIN_CONTEXT = 20  # alignment contexts seen fewer times use the pooled counts
MIN_SHARE = 0.02  # aligned spellings rarer than this per phone chunk are not tried
BRAHMIC = {"Deva", "Beng", "Guru", "Gujr", "Orya", "Taml", "Telu", "Knda", "Mlym",
           "Sinh"}  # abugidas spelled here: a vowel sign or virama needs a consonant
ABUGIDAS = {"Thai", "Laoo", "Khmr", "Mymr", "Ethi", "Tibt", "Hang", "Hira", "Kana"}
VOWEL_LETTERS = {"A", "AA", "I", "II", "U", "UU", "E", "EE", "AI", "O", "OO", "AU",
                 "AE", "AAE"}  # independent vowels, by Unicode name after "LETTER "


def _pool(alts):
    """Top alternatives plus the shortest one, so ranking can drop silent letters."""
    pool = alts[:MAX_ALTS]
    return pool + [s for s in [min(alts, key=len)] if s not in pool]


def context(prev, nxt):
    """Neighbouring phones as V (vowel), C (consonant) or # (word edge): V_C."""
    def kind(p):
        return "#" if p is None else "V" if is_vowel(p) else "C"
    return f"{kind(prev)}_{kind(nxt)}"


def _attach(phones):
    """Secondary marks split off by Target.prepare (l, ʲ) rejoined: lʲ."""
    out = []
    for p in phones:
        if p in ("ʲ", "ʷ") and out:
            out[-1] += p
        else:
            out.append(p)
    return tuple(out)


def _kind(c):
    """Brahmic character class: consonant, vowel (letter), sign, virama, nukta, mark
    (anusvara, candrabindu, visarga), other."""
    name = unicodedata.name(c, "")
    if "VOWEL SIGN" in name:
        return "sign"
    if "VIRAMA" in name or "AL-LAKUNA" in name:
        return "virama"
    if "NUKTA" in name:
        return "nukta"
    if any(m in name for m in ("ANUSVARA", "CANDRABINDU", "VISARGA")):
        return "mark"
    if "LETTER" in name:
        rest = name.split("LETTER ", 1)[1]
        vowel = rest in VOWEL_LETTERS or rest.startswith(("VOCALIC", "CANDRA", "SHORT"))
        return "vowel" if vowel else "consonant"
    return "other"


def _fits(prev, letters):
    """A Brahmic spelling stays well formed when `letters` follow `prev`: vowel signs,
    virama and nukta only right after a consonant (ि after इ or a sign is not text);
    no independent vowel after a virama (व्इ for वि) and no anusvara or visarga at
    the start or after a virama."""
    state = _kind(prev[-1]) if prev else "start"
    for c in letters:
        kind = _kind(c)
        if kind in ("sign", "virama", "nukta"):
            if state not in ("consonant", "nukta"):
                return False
            kind = "consonant" if kind == "nukta" else kind
        elif kind == "vowel" and state == "virama":
            return False
        elif kind == "mark" and state in ("start", "virama"):
            return False
        state = kind
    return True


def _letters(text):
    return all(unicodedata.category(c)[0] in "LM" for c in text)


class Alphabet:
    def __init__(self, code):
        script = code.split("-")[1]
        if script in ABUGIDAS:
            raise ValueError(f"{code}: {script} needs its own module, not alphabet.py")
        self.code, self.target = code, Target(code)
        self.brahmic = script in BRAHMIC
        rows = defaultdict(list)  # phone tuple -> spellings, map order
        base = MAP_DIR / f"{code}.csv"
        with open(base if base.exists() else DATA / code / "base.csv",
                  encoding="utf-8") as f:
            for r in csv.DictReader(f):
                phon = r["Phon"].replace("<ons>", "")  # Brahmic independent vowel आ
                key = tuple(segments(phon, keep_length=True))
                orth = unicodedata.normalize("NFC", r["Orth"])
                if key and _letters(orth) and orth not in rows[key]:
                    rows[key].append(orth)
        read_back = {}  # n-gram rows: their IPA came from epitran, so they read back

        self.lm = model(code)
        final = defaultdict(list)  # word-final only: how a gram reads alone
        if self.lm:  # drop letters the language does not write (French õ ʒ ö)
            seen = self.lm.letters({c for v in rows.values() for o in v
                                    for c in o.lower()})
            rows = defaultdict(list, {k: [o for o in v if set(o.lower()) <= seen]
                                      for k, v in rows.items()})
        self.readings = self._readings()  # (gram, context, phones): on -> ɔ̃ (French)
        medial = defaultdict(set)
        for gram, context, key in self.readings:
            if context != "alone":
                medial[gram].add(key)
                if gram not in rows[key]:
                    rows[key].append(gram)
                    read_back[gram, key] = True
        for gram, context, key in self.readings:  # German final e = ə; not лог = lok
            if context == "alone" and key not in medial[gram]:
                final[key].append(gram)
                read_back[gram, key] = True

        units = DATA / code / "units.txt"  # common syllables (scripts/make_units.py)
        for unit in units.read_text(encoding="utf-8").split() if units.exists() else []:
            key = forward(code, unit)  # read whole: usable anywhere in a word
            if key and unit not in rows[key]:
                rows[key].append(unit)
                read_back[unit, key] = True

        self.align = self._alignment()  # P(letters | phones) from WikiPron words
        for (key, ctx), costs in self.align.items():
            for orth, cost in costs.items():
                fits = _letters(orth) and (not self.lm or set(orth) <= seen)
                if fits and cost <= -math.log(MIN_SHARE) and ctx != "*":
                    spellings = (final if ctx.endswith("#") else rows)[key]
                    spellings += [orth] * (orth not in spellings)
        forward_many(code, [o for v in [*rows.values(), *final.values()] for o in v])

        def order(key, orth):
            ok = read_back.get((orth, key)) or forward(code, orth) == key
            plausible = -self.lm.logprob(orth) if self.lm else 0
            return self.prior(key, orth) + (not ok), plausible

        if self.brahmic:  # a consonant letter alone spells its consonant: a vowel
            for k, v in list(rows.items()):  # sign replaces the inherent vowel (বা)
                for o in v:
                    if len(k) > 1 and len(o) == 1 and _kind(o) == "consonant":
                        rows[k[:1]] += [o] * (o not in rows[k[:1]])
        if self.brahmic:  # a vowel sign never spells a consonant (ृ for r: थृएन)
            for k, v in [*rows.items(), *final.items()]:
                if k and not any(map(is_vowel, k)):
                    v[:] = [o for o in v if any(_kind(c) != "sign" for c in o)]
        self.table = {k: sorted(v, key=lambda o: order(k, o))  # stable: map order on
                      for k, v in rows.items() if v}  # ties; filtering may empty a key
        self.final = {k: sorted(v, key=lambda o: order(k, o))
                      for k, v in final.items()}
        singles = [k[0] for k in self.table if len(k) == 1]
        self.consonants = tuple(p for p in singles if not is_vowel(p))
        self.vowels = tuple(p for p in singles if is_vowel(p))
        self.max_len = max(len(k) for k in [*self.table, *self.final])
        self.known = {p for k in self.table for p in k}  # phones in any key
        self.singles = {k[0] for k in self.table if len(k) == 1}  # kʲ kept whole

    def _alignment(self):
        """{(phones, context): {letters: -log P(letters | phones, context)}} from
        data/<code>/align_*.tsv (scripts/make_align.py). Context is context() of the
        neighbouring phones (Russian lʲ: л in C_V, ль in V_C), backed off to the next
        phone alone (_C) and to all positions (*) when seen under MIN_CONTEXT times.
        English gets none (NO_ALIGN)."""
        if self.code in NO_ALIGN:
            return {}
        path, counts = DATA / self.code / ALIGN_FILE, defaultdict(Counter)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    key, n = tuple(r["phones"].split()), int(r["count"])
                    letters = unicodedata.normalize("NFC", r["letters"])
                    if n >= MIN_ALIGN:
                        ctx = r["context"]
                        for c in ("*", ctx, ctx[1:]):  # all, V_C, _C (next only)
                            counts[key, c][letters] += n
        return {k: {o: -math.log(n / sum(c.values())) for o, n in c.items()}
                for k, c in counts.items()
                if k[1] == "*" or sum(c.values()) >= MIN_CONTEXT}

    def prior(self, key, letters, ctx="*"):
        """-log P(letters | key, ctx), else pooled over contexts; 0 without
        alignment."""
        if not self.align:
            return 0.0
        costs = (self.align.get((key, ctx)) or self.align.get((key, ctx[1:]))
                 or self.align.get((key, "*"), {}))
        return costs.get(letters, UNSEEN)

    def _frames(self):
        """Stable vowel and consonant letters to read grams between: the most frequent
        pair that reads as itself (ala = a l a, lal = l a l), sonorants l n m first;
        not French/English e, which goes silent. Either may be None."""
        letters = [g for g in self.lm.ngrams() if len(g) == 1]
        reads = {g: forward(self.code, g) for g in letters}
        vowels = [g for g in letters if reads[g] and all(map(is_vowel, reads[g]))]
        consonants = [g for g in letters
                      if reads[g] and not any(map(is_vowel, reads[g]))
                      and (not self.brahmic or _kind(g) == "consonant")]  # not ं
        prefer = [("l",), ("n",), ("m",)]  # sonorants: no voicing assimilation (лог+т)
        consonants.sort(key=lambda g: prefer.index(reads[g]) if reads[g] in prefer
                        else len(prefer))
        for v in vowels:
            for c in consonants[:3]:
                vcv, cvc = v + c + v, c + v + c
                if (forward(self.code, vcv) == reads[v] + reads[c] + reads[v]
                        and forward(self.code, cvc) == reads[c] + reads[v] + reads[c]):
                    return v, c
        return None, None

    def _readings(self):
        """[(gram, context, phones)] for the language's frequent letter n-grams: saved
        in data/<code>/ngrams.tsv (scripts/make_ngrams.py), else read by epitran now."""
        saved = DATA / self.code / "ngrams.tsv"
        if saved.exists():
            with open(saved, encoding="utf-8") as f:
                return [(r["gram"], r["context"], tuple(r["phones"].split()))
                        for r in csv.DictReader(f, delimiter="\t")]
        out = []
        for gram in self.lm.ngrams() if self.lm else []:
            out += [(gram, ctx, key) for ctx, key in self._medial(gram).items()]
            alone = forward(self.code, gram)
            if alone:
                out.append((gram, "alone", alone))
        return out

    def _medial(self, gram):
        """How `gram` reads inside a word, between two vowels and between two
        consonants (French on = ɔ̃ only before a consonant), so word-edge effects
        (Russian final devoicing г = k, French silent final s in ons) do not leak into
        the table. Readings whose frame does not survive are skipped."""
        if not hasattr(self, "_frame"):
            self._frame = self._frames()
        out = {}
        for ctx, f in zip(("between_vowels", "between_consonants"), self._frame):
            if not f or gram.startswith(f) or gram.endswith(f):  # at+t+t merges (tat)
                continue
            edge = forward(self.code, f)
            whole = forward(self.code, f + gram + f)
            n = len(edge)
            if whole[:n] == edge and whole[-n:] == edge and len(whole) > 2 * n:
                out[ctx] = whole[n:-n]
        return out

    def phones(self, ipa):
        """IPA -> phones; phones absent from every table key are substituted."""
        out = []
        secondary = {m: m if (m,) in self.table else g for m, g in SECONDARY.items()}
        prepared = self.target.prepare(ipa, keep_length=True, keep=self.singles,
                                       secondary=secondary)
        for p in prepared:
            if p not in self.known:
                p = self.substitute("".join(c for c in p if c not in LENGTH))
            if p is not None:
                out.append(p)
        return out

    def substitute(self, p):
        """Known phone for `p`: itself, its first known fallback, or the nearest."""
        if p in self.known:
            return p
        if p in FALLBACK and not FALLBACK[p]:
            return None  # empty list: drop the phone (ʔ)
        known = [f for f in FALLBACK.get(p, []) if f in self.known]
        return known[0] if known else closest(p, self.consonants, self.vowels)

    def alternatives(self, phones, i):
        """[(length, spellings)] for each table key starting at phone i; the word-final
        table only for a key that ends the word. A phone that fits no key here (German
        ə only occurs in `ne`) takes its nearest single-phone spelling."""
        out = []
        for n in range(1, min(self.max_len, len(phones) - i) + 1):
            key = tuple(phones[i : i + n])
            alts = self.table.get(key, [])
            if i + n == len(phones):
                alts = alts + [a for a in self.final.get(key, []) if a not in alts]
            near = [a for v in self.near.get(key[0], []) if n == 1
                    for a in self.table[(v,)][:2] if a not in alts]  # French ə: eu
            if alts or near:
                out.append((n, (_pool(alts) if alts else []) + near))
        if not out:
            near = closest(phones[i], self.consonants, self.vowels)
            out.append((1, _pool(self.table[(near,)]) if near is not None else [""]))
        return out

    @cached_property
    def near(self):
        """Vowel -> its NEAR closest other vowels, by feature distance."""
        return {v: sorted((u for u in self.vowels if u != v),
                          key=lambda u: sub_cost(v, u))[:NEAR] for v in self.vowels}

    @cached_property
    def probes(self):
        """(letters, phones) appended so a prefix reads as inside a word (French
        bet = bɛ, but beta = bɛta): the stable vowel, and consonant + vowel, the
        better reading counting, since a vowel alone can form a digraph with the
        prefix (French pa + i = pai = pɛ, but pa + li = pali)."""
        if self.brahmic and self.lm:  # consonant + vowel sign: valid after anything
            return self._brahmic_probe()
        v, c = self._frames() if self.lm else (None, None)
        if not v:
            return (("", ()),)
        out = [(v, forward(self.code, v))]
        if c:
            out.append((c + v, forward(self.code, c) + forward(self.code, v)))
        return tuple(out)

    def _brahmic_probe(self):
        """A bare consonant letter ending the word (Hindi र = r): the prefix's last
        inherent ə is then read as a word-internal one (कम + र, not कम alone = kəm).
        Native recovery 0.48, against 0.39 adding consonant + vowel sign (रा, which
        deletes it: कमरा kəmraː) and 0.32 with no probe."""
        letters = [g for g in self.lm.ngrams() if len(g) == 1]
        cons = [g for g in letters if _kind(g) == "consonant"]
        signs = [g for g in letters if _kind(g) == "sign"]
        for c in cons:
            for v in signs:
                read, alone = forward(self.code, c + v), forward(self.code, c)
                if len(read) == 2 and read[0] == alone[0] and is_vowel(read[1]):
                    return ((c, alone[:1]),)
        return (("", ()),)

    @cached_property
    def silent(self):
        """Letters silent at the word end after a consonant, tried there (French
        ale = al, so bette = bɛt); none for Spanish, Russian."""
        v, c = self._frames() if self.lm else (None, None)
        if not (v and c):
            return [""]
        letters = [g for g in self.lm.ngrams() if len(g) == 1]
        frame = forward(self.code, v + c)
        return [""] + [g for g in letters if forward(self.code, v + c + g) == frame]

    def _sound(self, spellings, refs, probes):
        """Read-back distance of each spelling to its reference phones, the best over
        the probes appended (l + ʲ is read back as one phone lʲ)."""
        best = [math.inf] * len(spellings)
        for letters, extra in probes:
            reads = forward_many(self.code, [s + letters for s in spellings])
            for k, (ref, read) in enumerate(zip(refs, reads)):
                d = distance(_attach(tuple(ref) + extra), read, weighted=True)
                best[k] = min(best[k], d)
        return best

    def _settle(self, s, node):
        """Window: once more than 2 x WINDOW letters are unsettled, everything up to
        the last chunk boundary WINDOW letters back is read once and its cost frozen,
        so long input (unspaced sentences) is not re-read from the start each step."""
        prior, bounds, cut_l, cut_p, frozen = node
        if len(s) - cut_l <= 2 * WINDOW:
            return node
        done = [(l, p) for l, p in bounds if len(s) - l >= WINDOW]
        if not done:
            return node
        l, p = done[-1]
        frozen += self._sound([s[cut_l:l]], [self.phones_[cut_p:p]], self.probes)[0]
        return prior, tuple(b for b in bounds if b[0] > l), l, p, frozen

    def best(self, nodes, i, probes=(("", ()),), keep=None):
        """The `keep` cheapest of {spelling: node}: frozen cost + read-back distance of
        the unsettled letters to their phones (phones_[cut_p:i]) + letter, charlm and
        alignment costs. A node is (alignment cost, chunk bounds, cut_l, cut_p,
        frozen cost)."""
        items = [(s, self._settle(s, n)) for s, n in nodes.items()]
        sound = self._sound([s[n[2]:] for s, n in items],
                            [self.phones_[n[3]:i] for _, n in items], probes)
        end = not probes[0][0]

        def score(k):
            s, (prior, _, _, _, frozen) = items[k]
            plausible = -self.lm.logprob(s, end=end) if self.lm else len(s)
            cost = frozen + sound[k] + ALIGN * prior
            return cost + LETTER * len(s) + SURPRISAL * plausible, plausible
        ranked = sorted(range(len(items)), key=score)[: keep or BEAM]
        return dict(items[k] for k in ranked)

    def __call__(self, ipa):
        """One word of IPA -> spelling: beam search over table spellings, each prefix
        scored by how epitran reads it (context decides: French ente = ɑ̃t)."""
        phones = self.phones_ = tuple(self.phones(ipa))
        states = defaultdict(dict, {0: {"": (0.0, (), 0, 0, 0.0)}})
        for i in range(len(phones)):
            beam = states.pop(i, {})
            if not beam:
                continue  # unreachable: only longer keys spanned it
            beam = self.best(beam, i, self.probes) if i else beam
            for n, alts in self.alternatives(phones, i):
                key, nxt = tuple(phones[i : i + n]), states[i + n]
                ctx = context(phones[i - 1] if i else None,
                              phones[i + n] if i + n < len(phones) else None)
                cap = MAX_LETTERS * (i + n) + 2
                vowel_next = i + n < len(phones) and is_vowel(phones[i + n])
                for s, (c, bounds, cut_l, cut_p, frozen) in beam.items():
                    for a in alts:
                        if len(s) + len(a) > cap or (self.brahmic and (
                                not _fits(s, a) or vowel_next  # no sign after ्
                                and a and _kind(a[-1]) == "virama")):
                            continue
                        cost = c + self.prior(key, a, ctx)
                        if cost < nxt.get(s + a, (math.inf,))[0]:
                            bound = bounds + ((len(s) + len(a), i + n),)
                            nxt[s + a] = (cost, bound, cut_l, cut_p, frozen)
        if not states[len(phones)] and self.brahmic:  # no well-formed spelling
            self.brahmic = False
            try:
                return self(ipa)
            finally:
                self.brahmic = True
        done = self.best(states[len(phones)], len(phones))
        ends = {s + x: n for s, n in done.items() for x in self.silent
                if not self.brahmic or _fits(s, x)}
        out = next(iter(self.best(ends, len(phones), keep=1)))
        return unicodedata.normalize("NFC", self.target.finish(out))


@lru_cache(maxsize=None)
def converter(code):
    return Alphabet(code)


if __name__ == "__main__":
    conv = converter(sys.argv[1])
    for word in sys.argv[2:]:
        out = conv(word)
        print(word, out, "".join(forward(conv.code, out)))
