# -*- coding: utf-8 -*-
"""Offline learned-transliteration cache builder.

Run on any machine with Python <= 3.10 where fairseq builds:
    pip install ai4bharat-transliteration
    python offline_indicxlit.py reps_<model>_<N>langs_<S>sents_meta.json

Writes xlit_cache.json next to the meta file. Drop both into the notebook's working
directory and the v3 walkthrough picks the cache up automatically (candidate
'indicxlit_offline' in Part 8b)."""
import json, re, sys, os
from unidecode import unidecode

XLIT_LANGS = {
    "hin_Deva": "hi", "urd_Arab": "ur", "kas_Deva": "ks", "kas_Arab": "ks",
    "snd_Deva": "sd", "snd_Arab": "sd", "mar_Deva": "mr", "npi_Deva": "ne",
    "ben_Beng": "bn", "guj_Gujr": "gu", "pan_Guru": "pa", "tam_Taml": "ta",
    "tel_Telu": "te", "kan_Knda": "kn", "mal_Mlym": "ml", "sin_Sinh": "si",
}

def main():
    meta_path = sys.argv[1]
    meta = json.load(open(meta_path, encoding="utf-8"))
    text, ids = meta["text"], meta["ids"]

    from ai4bharat.transliteration import XlitEngine
    langs = sorted(set(XLIT_LANGS[c] for c in text if c in XLIT_LANGS))
    eng = XlitEngine(langs, src_script_type="indic", beam_width=4, rescore=False)

    cache = {}
    def word(w, lg):
        key = (w, lg)
        if key not in cache:
            try:
                r = eng.translit_word(w, lang_code=lg, topk=1)
                cache[key] = ((r.get(lg) or r.get("en"))[0] if isinstance(r, dict) else r[0]).lower()
            except Exception:
                cache[key] = unidecode(w).lower()
        return cache[key]

    romanized = {}
    for code, sents in text.items():
        lg = XLIT_LANGS.get(code)
        if lg is None:
            romanized[code] = [unidecode(s).lower() for s in sents]
        else:
            romanized[code] = [" ".join(word(w, lg) for w in re.findall(r"\w+", s))
                               for s in sents]
        print("done:", code)

    out = os.path.join(os.path.dirname(meta_path) or ".", "xlit_cache.json")
    json.dump({"ids": ids, "romanized": romanized}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False)
    print("wrote:", out)

if __name__ == "__main__":
    main()
