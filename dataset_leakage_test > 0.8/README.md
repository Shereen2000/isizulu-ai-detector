# Class-Level Leakage Detection — Results Guide

## What This Analysis Does

This audit checks whether the same (or near-identical) text appears in more than one
dataset split (train, eval, test), analysed **independently per class** (human vs machine).

Splitting by class before running leakage detection answers a more precise question than
a combined check: *within each class, did the same source documents bleed across splits?*

---

## Methodology

| Parameter | Value |
|---|---|
| Near-duplicate threshold | Jaccard similarity >= 80% |
| Shingling | Character 5-grams |
| MinHash permutations | 128 |
| Exact duplicate method | MD5 hash of lowercased, stripped text |

**Jaccard similarity** measures how much two texts share in terms of character n-gram overlap.
A score of 1.0 means the texts are identical; 0.8 means 80% of their character patterns are shared.

---

## Output Files

| File | What it contains |
|---|---|
| `human/train.jsonl` | All label 0 (human-written) samples from train |
| `human/eval.jsonl` | All label 0 samples from eval |
| `human/test.jsonl` | All label 0 samples from test |
| `machine/train.jsonl` | All label 1 (machine-written) samples from train |
| `machine/eval.jsonl` | All label 1 samples from eval |
| `machine/test.jsonl` | All label 1 samples from test |
| `human/exact_leak_*.csv` | Exact duplicate pairs found within the human class across splits |
| `human/near_leak_*.csv` | Near-duplicate pairs (>= 80% similar) within the human class |
| `machine/exact_leak_*.csv` | Exact duplicate pairs within the machine class |
| `machine/near_leak_*.csv` | Near-duplicate pairs within the machine class |
| `class_leakage_summary.json` | Aggregated counts per class and split pair |

---

## How to Interpret the Results

### Exact Duplicates (100% identical)

These are the most serious form of leakage. The same text appears in two different splits
under the same class label.

| Count | Interpretation |
|---|---|
| 0 | Clean — no leakage |
| Low (1–20) | Minor leakage — likely caused by duplicates in the original source corpus |
| High (20+) | Significant leakage — source data was not deduplicated before splitting |

### Near-Duplicates (>= 80% Jaccard)

These are texts that are not identical but share most of their character patterns —
typically the same document with minor edits, truncations, or formatting differences.

| Count | Interpretation |
|---|---|
| 0 | Clean |
| Low | Acceptable — minor overlap from similar source documents |
| High | Suggests source documents were not diverse enough, or splitting was not stratified by document |

---

## What the Results Mean for Model Performance

### If leakage is found in the HUMAN class only
The model was trained and tested on near-identical human texts. This means the model
may have memorised specific human writing patterns from the source documents rather than
learning general human linguistic features. Human recall may be artificially inflated.

### If leakage is found in the MACHINE class only
The machine-generated texts (produced via back-translation) share source documents across
splits. Since machine texts are derived from human texts, this is more likely when the
same human source document was used to generate machine samples in both train and test.
Machine recall may be artificially inflated.

### If leakage is found in BOTH classes
The source corpus itself contained duplicates that propagated into all splits. This is
the most common cause and suggests deduplication should have been applied to the raw
corpus before class generation and splitting.

### If NO leakage is found
The splits are clean and independent. Model performance metrics (accuracy, F1, ROC-AUC)
reflect genuine generalisation to unseen data, not memorisation of training examples.

---

## Implications for the Research Report

- **Disclose** any leakage found, even if minor. Reviewers expect this analysis.
- **Quantify** the leakage as a percentage of the test set (e.g. 18/2022 = 0.89%).
- **Argue** whether the leakage is likely to have materially affected results. For small
  percentages (< 1%), the impact on aggregate metrics like F1 is negligible.
- **Recommend** deduplication as a preprocessing step for future work.
- A model that still achieves 99%+ F1 even with disclosed minor leakage is a stronger
  result than one where leakage is hidden — transparency strengthens the contribution.

---

## CSV Column Reference

### `exact_leak_*.csv`
| Column | Meaning |
|---|---|
| `train_idx` | Index of the sample in the train split |
| `eval_idx` / `test_idx` | Index of the matching sample in eval/test |
| `text_snippet` | First 120 characters of the text |

### `near_leak_*.csv`
| Column | Meaning |
|---|---|
| `train_idx` | Index in train |
| `eval_idx` / `test_idx` | Index in eval/test |
| `jaccard_similarity` | Similarity score (0.0–1.0) |
| `train_snippet` | First 100 characters of the train text |
| `eval_snippet` / `test_snippet` | First 100 characters of the other split text |
