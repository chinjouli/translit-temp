# p2g transliteration demo

Transliterate words into another script via IPA: a source spelling (read by [epitran](https://github.com/dmort27/epitran), English via [Flite](https://github.com/festvox/flite)) or raw IPA goes to Hangul, kana, pinyin, bopomofo, or any of ~70 alphabets and abjads.

Run locally: `streamlit run streamlit_app.py`. On Streamlit Community Cloud, pick Python 3.11+; `packages.txt` installs `flite1-dev` and the app compiles Flite's `lex_lookup` (`lex_lookup_main.c`, CMU licence) on first start. Feedback goes to Formspree: set `FORMSPREE_ID` in the app's Secrets (or the environment); without it, reviewers can still download their feedback as TSV.

Spelling models are trained on [WikiPron](https://github.com/CUNY-CL/wikipron) data, scraped from Wiktionary (CC BY-SA 3.0). epitran is vendored with local map additions (MIT).
