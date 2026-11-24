"""Streamlit demo of p2g/translit.py: IPA or a source orthography -> a target script."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "p2g", HERE / "p2g", HERE / "epitran"):  # local, deployed
    if p.exists():
        sys.path.insert(0, str(p))
os.environ.setdefault("APERTUS_TOKENIZER", str(HERE / "apertus_tokenizer.json"))

import epitran  # noqa: E402
import source  # noqa: E402
from engine import DATA  # noqa: E402
from translit import ALIASES, converter, reader  # noqa: E402

IPA = "IPA (type IPA directly)"
BUILT = ("align_wikipron.tsv", "roman.csv", "syllables.txt")  # as translit.converter
TARGETS = list(ALIASES) + sorted(
    d.name for d in DATA.iterdir()
    if d.name not in ALIASES.values() and any((d / f).exists() for f in BUILT))
MAPS = Path(epitran.__file__).parent / "data/map"
HIDDEN = {"cmn-Bopo", "cmn-Latn"}  # superseded by the "2" maps
SOURCES = [IPA, "eng-Latn"] + sorted(p.stem for p in MAPS.glob("*.csv")
                                     if p.stem not in HIDDEN)


@st.cache_resource
def flite():
    """Put Flite's lex_lookup on PATH, compiling it against flite1-dev if missing."""
    if shutil.which("lex_lookup"):
        return
    bin_dir = Path(tempfile.gettempdir()) / "p2g-bin"
    bin_dir.mkdir(exist_ok=True)
    subprocess.run(["gcc", "-O2", "-I/usr/include/flite", HERE / "lex_lookup_main.c",
                    "-o", bin_dir / "lex_lookup", "-lflite_cmulex", "-lflite_usenglish",
                    "-lflite", "-lm"], check=True)
    os.environ["PATH"] = f"{bin_dir}:{os.environ['PATH']}"


@st.cache_resource(show_spinner="Building the target model (first use only)...")
def target(name):
    return converter(name)


def run(src, tgt, words, stress):
    conv, rows = target(tgt), []
    for word in words:
        try:
            ipa = word if src == IPA else source.to_ipa(reader(src), word, stress)
            rows.append({"word": word, "IPA": ipa, "output": conv(ipa)})
        except Exception as e:  # show the failure, keep the other words
            rows.append({"word": word, "IPA": "", "output": f"error: {e}"})
    return rows


st.set_page_config(page_title="p2g transliteration")
st.title("p2g: transliterate via IPA")
st.write("Pick a source (raw IPA, or a spelling read by epitran; English by Flite) "
         "and a target script. Words are separated by spaces.")
flite()
c1, c2 = st.columns(2)
src = c1.selectbox("Source", SOURCES, index=1)
tgt = c2.selectbox("Target", TARGETS, index=TARGETS.index("kor"))
text = st.text_input("Words", "computer hotel coffee")
stress = st.checkbox("English: aspirate stops starting stressed syllables "
                     "(off = word-initial only)", value=True)
if text.split():
    st.dataframe(run(src, tgt, text.split(), stress), hide_index=True, width="stretch")
