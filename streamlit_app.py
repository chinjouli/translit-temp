"""Streamlit demo of p2g/translit.py: IPA or a source orthography -> a target script."""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests
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
HIDDEN = {"cmn-Bopo", "cmn-Latn"}  # superseded by the "2" maps
BUILT = ("align_wikipron.tsv", "roman.csv", "syllables.txt")  # as translit.converter
SKIP = HIDDEN | set(ALIASES.values())
TARGETS = list(ALIASES) + sorted(d.name for d in DATA.iterdir() if d.name not in SKIP
                                 and any((d / f).exists() for f in BUILT))
COLS = ["source", "target", "word", "IPA", "output", "correct", "comment"]
EDIT = {"correct": st.column_config.TextColumn("✏️ correct spelling"),
        "comment": st.column_config.TextColumn("✏️ comment")}
MAPS = Path(epitran.__file__).parent / "data/map"
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


def version():
    """Deployed commit, to tie feedback to the code that produced it."""
    git = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    return git.stdout.strip() or "unknown"


def form_id():
    """Formspree form id from Streamlit secrets or the environment, else None."""
    try:
        return st.secrets["FORMSPREE_ID"]
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return os.environ.get("FORMSPREE_ID")


def submit(reviewer, rows):
    """POST one batch to Formspree; returns an error message or None."""
    payload = {"_subject": f"p2g feedback: {len(rows)} row(s), {reviewer or 'anon'}",
               "reviewer": reviewer, "version": version(), "count": len(rows),
               "rows": rows.to_csv(sep="\t", index=False)}
    r = requests.post(f"https://formspree.io/f/{form_id()}", json=payload,
                      headers={"Accept": "application/json"}, timeout=20)
    return None if r.ok else f"{r.status_code}: {r.text[:300]}"


def run(src, tgt, words, stress):
    conv, rows = target(tgt), []
    for word in words:
        try:
            ipa = word if src == IPA else source.to_ipa(reader(src), word, stress)
            rows.append({"word": word, "IPA": ipa, "output": conv(ipa)})
        except Exception as e:  # show the failure, keep the other words
            rows.append({"word": word, "IPA": "", "output": f"error: {e}"})
    return rows


st.set_page_config(page_title="p2g transliteration", layout="wide")
st.title("p2g: transliterate via IPA")
flite()
ss = st.session_state
ss.setdefault("fb", pd.DataFrame(columns=COLS))  # feedback basket
ss.setdefault("fb_ver", 0)  # bumped to reload the basket editor with new rows
ss.setdefault("res_ver", 0)  # bumped to clear the results editor after adding
ss.setdefault("sent_at", 0.0)
if note := ss.pop("toast", None):
    st.toast(note)

st.markdown("""**How to give feedback**
1. Pick a **source** (a spelling system read by epitran, English by Flite, or raw IPA)
   and a **target** script, and type some words separated by spaces.
2. In the results, type the **correct spelling** and/or a **comment** on any wrong row,
   then click **Add commented rows to feedback**. Repeat with other words or languages.
3. Check **Your feedback** at the bottom, add your name and languages, and click
   **Submit feedback** once at the end (or **Download TSV** and email it).""")

c1, c2 = st.columns(2)
src = c1.selectbox("Source", SOURCES, index=1)
tgt = c2.selectbox("Target", TARGETS, index=TARGETS.index("kor"))
text = st.text_input("Words", "computer hotel coffee")
stress = st.checkbox("English: aspirate stops starting stressed syllables "
                     "(off = word-initial only)", value=True)

if text.split():
    res = pd.DataFrame(run(src, tgt, text.split(), stress))
    res = res.assign(correct="", comment="")
    res = st.data_editor(res, hide_index=True, width="stretch", column_config=EDIT,
                         disabled=["word", "IPA", "output"],
                         key=f"res-{src}-{tgt}-{stress}-{text}-{ss.res_ver}").fillna("")
    if st.button("Add commented rows to feedback"):
        new = res[(res.correct.str.strip() != "") | (res.comment.str.strip() != "")]
        new = new.assign(source=src, target=tgt)[COLS]
        key = ["source", "target", "word", "IPA", "output"]  # re-adding replaces
        ss.fb = pd.concat([ss.get("fb_now", ss.fb), new], ignore_index=True)
        ss.fb = ss.fb.drop_duplicates(key, keep="last", ignore_index=True)
        ss.fb_ver, ss.res_ver = ss.fb_ver + 1, ss.res_ver + 1
        ss.toast = f"Added {len(new)} row(s)" if len(new) else "No commented rows"
        st.rerun()

st.subheader("Your feedback")
if msg := ss.pop("msg", None):
    st.success(msg)
ss.fb_now = st.data_editor(ss.fb, num_rows="dynamic", hide_index=True, width="stretch",
                           column_config=EDIT, key=f"fb-{ss.fb_ver}",
                           disabled=["source", "target", "word", "IPA", "output"])
reviewer = st.text_input("Your name and languages (optional)")
c1, c2 = st.columns(2)
rows = ss.fb_now.fillna("")
if c1.button("Submit feedback", type="primary", disabled=rows.empty or not form_id()):
    if time.time() - ss.sent_at < 30:
        st.warning("Please wait 30 seconds between submissions.")
    elif err := submit(reviewer, rows):
        st.error(f"Submission failed ({err}). Download the TSV and email it instead.")
    else:
        ss.sent_at, ss.fb_ver = time.time(), ss.fb_ver + 1
        ss.fb, ss.msg = pd.DataFrame(columns=COLS), f"Thanks! Sent {len(rows)} row(s)."
        st.rerun()
c2.download_button("Download TSV", rows.to_csv(sep="\t", index=False),
                   "p2g_feedback.tsv", disabled=rows.empty)
