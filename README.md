# Cross-script entity matching

Code and data for measuring and localizing cross-script entity matching failures in
instruction-tuned language models: deciding whether two surface forms, written in
different scripts, refer to the same person.

The task is one question. Given a pair such as `महात्मा गांधी` and `बापू`, or the same
names written `Mahatma Gandhi` and `Bapu`, does the model recognize that both name the
same person? Each identity pair is rendered in four script configurations and asked in
two prompt languages, with a matched non-matching pair as a control.

## What is here

```
corpora/                 the two evaluation corpora, manually verified
  mr/                    601 Marathi identity pairs and their hard negatives
  hi/                    601 Hindi identity pairs and their hard negatives
pipeline/                the experiment, stages 00-13
corpus_construction/     building a corpus from Wikidata
analysis/                table generation and verification
```

Results are not included. Every number is reproducible from the pipeline.

## The corpora

Each row is one identity pair: two names for the same person, each in both a
Devanagari and a Latin form.

| column | meaning |
|---|---|
| `fact_id` | Wikidata QID, prefixed |
| `name_a_dev`, `name_a_lat` | first name, native script and Latin |
| `name_b_dev`, `name_b_lat` | second name, native script and Latin |
| `relation_tier` | how the alias relation is attested in Wikidata |
| `origin` | `native` or `foreign`, by country of citizenship |
| `generated_lat` | whether the Latin form was generated rather than attested |

`hard_negative_map.csv` pairs each fact with a non-matching fact of the same type, so a
model cannot score well by answering "same person" to everything. Every fact has exactly
one negative, no negative is a self-pair, and no item is reused as a negative more than
five times.

Both corpora were checked by hand, item by item. This matters more than it sounds:
automated cross-script name matching agreed with human labels only about 75 percent of
the time in our testing, so the pairs cannot be validated by transliteration distance.
No Indic transliterator was used to build or repair either corpus, because the
transliterators are themselves under test in stage 12. Latin forms come from Wikidata
labels and aliases.

Two scripts in `corpus_construction/` were tried and rejected, and carry headers saying
so: `repair_pairs.py`, which damages a verified corpus by maximising string similarity,
and `pair_selection.py`, which selects pairs by transliteration distance at 59 to 68
percent precision. They are kept because the reasons they fail are not obvious in
advance. Do not run either against the corpora here.

## Running the experiment

```bash
pip install -r pipeline/requirements_a100.txt
```

A single 80GB GPU is enough. Stages are resumable and skip completed work unless
`--force` is passed.

```bash
cd pipeline
python run_languages.py --languages mr --model qwen --corpus-root ../corpora \
  --stages 00 01 02 03 04 05 06 07 08 09 10 11 12
```

Swap `--languages hi` and `--model gemma` or `--model llama` for the other arms. Stage
13 is separate:

```bash
python 13_jacobian_script_pair.py --run-dir results/qwen_mr --model qwen
```

Expect roughly three hours per model on an A100, dominated by stages 04 and 07.

### Stages

| stage | what it does |
|---|---|
| 00 | audit the corpus, build train/validation/test splits |
| 01 | score the behavioural factorial, build the knowledge gate |
| 02 | cache hidden states at four token positions |
| 03 | probes, retrieval, tokenization, failure prediction |
| 04 | activation patching by site and layer, with controls |
| 05 | decompose the effect into attention and MLP |
| 06 | screen and confirm individual attention heads |
| 07 | attempted repair: correction vector and prompt anchoring |
| 08 | collect results into a dashboard |
| 09 | gradient lens on the output margin |
| 10 | name surprisal and tokenizer fertility |
| 11 | swap the two names to separate position from name |
| 12 | replace names with five automatic romanizations |
| 13 | gradient lens taken in both script conditions |

### Adding a language

Add three question paraphrases to `pipeline/question_templates.json` and place a corpus
at `corpora/<lang>/`. Nothing else is language-specific. A language is marked
`reviewed` only once a native speaker has checked the templates; until then every stage
prints a warning, because a poor template produces a null result indistinguishable from
a model that cannot do the task.

### Rebuilding a corpus

```bash
pip install -r corpus_construction/requirements.txt
```

The scripts in `corpus_construction/` query Wikidata for entities with two attested
names, band them by sitelink count, and produce a review sheet for a human to check.
The shipped corpora are the output of that process plus manual verification of every
item.

## Checking the numbers

`analysis/paper_tables.py` regenerates tables from a results directory and compares them
against a LaTeX source, reporting any numeric literal that differs:

```bash
python analysis/paper_tables.py --check
python analysis/paper_tables.py --emit
```

`pipeline/audit_runs.py` reports what a run actually produced, which is not the same
question as whether it exited cleanly. A stage can write every output and then fail, and
it can exit cleanly having written a file with no rows for one of the prompt languages.

```bash
python pipeline/audit_runs.py --results results
```

## Two notes on reading the results

**Response bias is separated from sensitivity.** Positive-pair accuracy alone cannot
distinguish a model that knows the answer from one that says "yes" to everything. One of
the three models tested answers "same person" to over 90 percent of positive pairs and
also to three-quarters of the non-matching ones. Sensitivity is therefore reported as
d′ with a log-linear correction, and the causal gate can be read on either polarity via
`--gate` and `--gate-polarity`, so a model whose failures are the negative pairs is
still testable.

**`flip_rate` in the patching tables** holds the fraction of items correct *after* the
intervention, not the fraction whose answer changed. The two coincide only where the
baseline is wrong for every item. Use `baseline_correct` with `patched_correct` instead.

## Models

Qwen2.5-7B-Instruct under Apache 2.0, Gemma-2-2b-it under the Gemma Terms of Use, and
Llama-3.1-8B-Instruct under the Llama 3.1 Community License. No model weights are
redistributed here.

## Licence

Code is released under the MIT License. The corpora derive from Wikidata and carry
Wikidata's CC0 dedication.
