"""Instruction-following pilot. Run before committing GPU time to any language.

Llama-3.1-8B scored d-prime 0.02 in Marathi: it could not do the task at all, and that
only became clear after a full overnight run. This checks the same thing on 20 items per
language in a few minutes, and reports which language-by-model cells are worth running.

A cell passes if negatives are answered correctly often enough that the model is
evidently reading the question, and if sensitivity is meaningfully above zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent

# The question templates. Each is rendered with the two names in brackets.
TEMPLATES = {
    "en": "Do [{a}] and [{b}] refer to the same person?",
    "mr": "[{a}] आणि [{b}] हे एकच व्यक्ती आहेत का?",
    "hi": "क्या [{a}] और [{b}] एक ही व्यक्ति हैं?",
    "ne": "के [{a}] र [{b}] एउटै व्यक्ति हुन्?",
    "ta": "[{a}] மற்றும் [{b}] ஒரே நபரா?",
    "te": "[{a}] మరియు [{b}] ఒకే వ్యక్తియేనా?",
    "kn": "[{a}] ಮತ್ತು [{b}] ಒಬ್ಬರೇ ವ್ಯಕ್ತಿಯೇ?",
    "ml": "[{a}] ഉം [{b}] ഉം ഒരേ വ്യക്തിയാണോ?",
    "bn": "[{a}] এবং [{b}] কি একই ব্যক্তি?",
    "gu": "શું [{a}] અને [{b}] એક જ વ્યક્તિ છે?",
    "pa": "ਕੀ [{a}] ਅਤੇ [{b}] ਇੱਕੋ ਵਿਅਕਤੀ ਹਨ?",
    "ru": "Относятся ли [{a}] и [{b}] к одному и тому же человеку?",
    "ar": "هل يشير [{a}] و [{b}] إلى نفس الشخص؟",
    "zh": "[{a}] 和 [{b}] 是同一个人吗?",
    "ko": "[{a}]와(과) [{b}]는 같은 사람입니까?",
    "el": "Αναφέρονται τα [{a}] και [{b}] στο ίδιο πρόσωπο;",
    "he": "האם [{a}] ו-[{b}] מתייחסים לאותו אדם?",
    "th": "[{a}] และ [{b}] เป็นคนเดียวกันหรือไม่?",
    "ja": "[{a}] と [{b}] は同一人物ですか?",
    "sr": "Да ли се [{a}] и [{b}] односе на исту особу?",
    "ur": "کیا [{a}] اور [{b}] ایک ہی شخص ہیں؟",
    "vi": "[{a}] và [{b}] có phải là cùng một người không?",
    "tr": "[{a}] ve [{b}] aynı kişi mi?",
    "cs": "Označují [{a}] a [{b}] tutéž osobu?",
    "yo": "Ṣé [{a}] àti [{b}] jẹ́ ẹni kan náà?",
    "is": "Eiga [{a}] og [{b}] við sömu manneskju?",
    "id": "Apakah [{a}] dan [{b}] adalah orang yang sama?",
    "sw": "Je, [{a}] na [{b}] ni mtu yule yule?",
}

INSTRUCTION = "\n\nAnswer only A or B. {yes} = yes. {no} = no."


def dprime(hit: float, fa: float, n: int) -> float:
    e = 0.5 / max(n, 1)
    return float(norm.ppf(np.clip(hit, e, 1 - e)) - norm.ppf(np.clip(fa, e, 1 - e)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HuggingFace model id")
    ap.add_argument("--corpora", type=Path, default=HERE / "out" / "corpora")
    ap.add_argument("--languages", nargs="+", default=None)
    ap.add_argument("--items", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=HERE / "out" / "pilot")
    ap.add_argument("--min-negative", type=float, default=0.80,
                    help="negatives must be answered correctly at least this often")
    ap.add_argument("--min-dprime", type=float, default=0.40)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    device = next(model.parameters()).device
    a_id = tok.encode("A", add_special_tokens=False)
    b_id = tok.encode("B", add_special_tokens=False)
    if len(a_id) != 1 or len(b_id) != 1:
        raise SystemExit(f"labels are not single tokens for {args.model}: {a_id}, {b_id}")
    a_id, b_id = a_id[0], b_id[0]

    keys = args.languages or sorted(d.name for d in args.corpora.iterdir() if d.is_dir())
    rows, detail = [], []

    for key in keys:
        cpath = args.corpora / key / "corpus_selected.csv"
        npath = args.corpora / key / "hard_negative_map.csv"
        if not cpath.exists():
            print(f"[{key}] no corpus, skipped"); continue
        if key not in TEMPLATES:
            print(f"[{key}] no question template, skipped"); continue

        corpus = pd.read_csv(cpath).head(args.items)
        negs = pd.read_csv(npath) if npath.exists() else pd.DataFrame()
        neg_of = dict(zip(negs.get("fact_id", []), negs.get("negative_fact_id", [])))
        by_id = corpus.set_index("fact_id")

        prompts, meta = [], []
        for prompt_lang in (key, "en"):
            for _, r in corpus.iterrows():
                for truth in (1, 0):
                    if truth == 1:
                        b_name = r["name_b_dev"]
                    else:
                        nid = neg_of.get(r["fact_id"])
                        if nid is None or nid not in by_id.index:
                            continue
                        b_name = by_id.loc[nid, "name_b_dev"]
                    for yes in ("A", "B"):
                        q = TEMPLATES[prompt_lang].format(a=r["name_a_dev"], b=b_name)
                        q += INSTRUCTION.format(yes=yes, no="B" if yes == "A" else "A")
                        prompts.append(tok.apply_chat_template(
                            [{"role": "user", "content": q}],
                            tokenize=False, add_generation_prompt=True))
                        meta.append({"lang": key, "prompt_lang": prompt_lang,
                                     "fact_id": r["fact_id"], "truth": truth, "yes": yes})

        margins = []
        with torch.inference_mode():
            for i in range(0, len(prompts), args.batch_size):
                enc = tok(prompts[i:i + args.batch_size], return_tensors="pt",
                          padding=True, truncation=True, max_length=384).to(device)
                lg = model(**enc, use_cache=False).logits[:, -1].float()
                for row, m in zip(lg, meta[i:i + args.batch_size]):
                    yid, nid_ = (a_id, b_id) if m["yes"] == "A" else (b_id, a_id)
                    margins.append(float(row[yid] - row[nid_]))

        df = pd.DataFrame(meta)
        df["margin"] = margins
        df["correct"] = ((df.margin > 0) == (df.truth == 1)).astype(int)
        detail.append(df)

        for plang, g in df.groupby("prompt_lang"):
            pos = g[g.truth == 1].correct.mean()
            neg = g[g.truth == 0].correct.mean()
            d = dprime(pos, 1 - neg, g.fact_id.nunique())
            ok = (neg >= args.min_negative) and (d >= args.min_dprime)
            rows.append({"lang": key, "prompt_lang": plang,
                         "positives": round(pos, 3), "negatives": round(neg, 3),
                         "dprime": round(d, 3), "n_items": g.fact_id.nunique(),
                         "usable": int(ok)})
            flag = "OK " if ok else "DROP"
            print(f"  {flag} {key}/{plang}: pos {pos:.2f} neg {neg:.2f} d' {d:+.2f}")

    if not rows:
        raise SystemExit("no language produced results")
    summary = pd.DataFrame(rows)
    tag = args.model.split("/")[-1]
    summary.to_csv(args.out / f"pilot_{tag}.csv", index=False)
    pd.concat(detail).to_csv(args.out / f"pilot_{tag}_detail.csv", index=False)

    print()
    print(summary.to_string(index=False))
    usable = sorted(summary[summary.usable == 1].lang.unique())
    dropped = sorted(set(summary.lang) - set(usable))
    print(f"\nusable ({len(usable)}): {', '.join(usable)}")
    if dropped:
        print(f"drop for this model ({len(dropped)}): {', '.join(dropped)}")
    (args.out / f"usable_{tag}.json").write_text(json.dumps(usable, indent=1))


if __name__ == "__main__":
    main()
