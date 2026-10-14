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
python run_languages.py --languages mr --model qwen --corpus-root ../corpora
```

Swap `--languages hi` and `--model gemma` or `--model llama` for the other arms.
Every stage runs by default, including stage 13; `--stages` narrows that.

A stage run by hand must be given `--contexts`, because its parser defaults to
`mr en`. Running stage 13 on a Hindi corpus without it silently scores Hindi names
under Marathi prompts, and the output looks well-formed. The runner passes the flag
for you, which is why it is the recommended entry point.

```bash
python 13_jacobian_script_pair.py --run-dir results/qwen_hi --model qwen \
  --contexts hi en
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

## The failure-conditioned mediator (stage 14)

`14_failure_mediator.py` asks whether a low-rank subspace of the second-entity state,
rather than the whole state, carries the repair:

    h_{D<-L}^U  =  h_D  +  U U^T (h_L - h_D)

`U` is learned through the frozen model with a failure-conditioned objective: repair the
gated items, and leave everything that does not need repair alone. Standard DAS is run
with the same parametrisation and the closure loss alone, so any gap between them comes
from the conditioning and the specificity penalty rather than from the fit. Untrained
baselines at the same rank and layer are a random subspace, PCA of the donor-minus-native
differences, the mean difference, the probe direction, the Jacobian direction, and the
full residual patch as an approximate upper bound.

It fits on train and evaluates on validation. The test split is refused unless
`--allow-test` is given, which exists for one frozen evaluation and nothing before it.
The defaults hold prompt template 2 out of fitting entirely, so the reported numbers are
leave-one-template-out.

Run it in two phases; each prints its cost in forward passes before starting:

```bash
# phase 1: sweep the layer at one rank
python pipeline/run_languages.py --stages 14 --languages mr --model qwen

# phase 2: sweep the rank at the layer phase 1 selected on validation
python pipeline/14_failure_mediator.py --run-dir results/qwen_mr --model qwen \
    --contexts mr en --layers 16 --ranks 1 2 4 8
```

Stage 14 is not in the default stage list. It costs roughly 100,000 forward-backward
passes per phase, which is not something an overnight run should pick up by accident.

Two things the script protects you from. Resuming into a directory whose stored
settings differ from the current arguments is refused, because `mediator_items.csv` is
append-only and mixing settings would corrupt the summary; use `--force` or a different
run directory. And `--precision-check` measures the bfloat16 floor on closure: a rank-1
subspace spanning the donor-minus-native difference is algebraically the whole-state
patch, so the margin gap between them is quantisation and nothing else.

One caveat to read before trusting a specificity number: the `robust` control group
(items answered correctly in both scripts) is nearly empty in native-script contexts,
because a model that fails almost every Devanagari positive has almost no such item.
Qwen Marathi validation has none. Specificity there rests on the `negatives` group, the
same items at the other polarity, and on `never`, items wrong in both scripts. The
script prints every group size and warns when a keep group is empty.

`pipeline/test_14_failure_mediator.py` checks the subspace algebra, that a gradient
reaches `U`, that every derived subspace is orthonormal at the requested rank, and that
the item groups are disjoint. It needs no GPU and no model weights:

```bash
python pipeline/test_14_failure_mediator.py --run-dir ../results_mr/qwen_mr
```

## Checking the numbers

`analysis/paper_tables.py` regenerates fourteen of the paper's tables from the result
files and compares them against the LaTeX source, reporting any numeric literal that
differs. Results may sit in more than one root; pass `--results` once per root. The
Marathi and Hindi runs both have to be present, because the paper's Hindi rows come
from the Hindi run:

```bash
python analysis/paper_tables.py --check --results ../results_mr --results ../results_hi \
                                --paper ../paper/final_draft.tex
python analysis/paper_tables.py --emit  --results ../results_mr --results ../results_hi \
                                --out tables_generated
```

`RESULTS_DIR` (path-separator-separated) and `PAPER_TEX` are accepted in place of the
flags. The fifteen tables the script does not regenerate are listed at the end of its
output with the reduction that reproduces each; they were verified by hand.

`analysis/dprime_bootstrap.py` recomputes the sensitivity gap and its interval,
resampling items rather than prompts, because prompts derived from one item are not
independent. Every pooled rate uses the log-linear estimator `(x + 0.5) / (n + 1)`.

```bash
python analysis/dprime_bootstrap.py --results results_mr --contexts mr en
python analysis/dprime_bootstrap.py --results results_hi --contexts hi en
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
