# Class-Level Leakage & Vocabulary Stability — Results Guide

## What This Analysis Does

Two complementary checks, both run independently per class (human vs machine):

**1. Leakage detection** — checks whether the same (or near-identical) text appears
across splits (train, eval, test). Exact duplicates and near-duplicates at five thresholds.

**2. Within-class vocabulary stability** — checks whether the top 100 most frequent
words in each class's training split maintain similar rates in eval and test. If they do,
the class vocabulary is stable across splits. If they shift, there is domain drift.

---

## Methodology

| Parameter | Value |
|---|---|
| Thresholds tested | [0.5, 0.6, 0.7, 0.8, 0.9] |
| Shingling | Character 5-grams |
| MinHash permutations | 128 |
| Exact duplicate method | MD5 hash of lowercased, stripped text |
| Top words tracked (vocab) | 100 per class |
| Min word frequency (vocab) | 5 occurrences in train |
| Significance test (vocab) | Chi-squared (α = 0.05) |

---

## Folder Structure

```
dataset_leakage_test/
├── human/
│   ├── train.jsonl                     ← all label 0 samples from train
│   ├── eval.jsonl                      ← all label 0 samples from eval
│   ├── test.jsonl                      ← all label 0 samples from test
│   ├── exact_leak_*.csv                ← exact duplicates (threshold-independent)
│   ├── vocab_stability.csv             ← within-class word rate stability
│   └── results/
│       ├── threshold_0.5/
│       │   ├── intra_*_near_dups.csv   ← intra-split near-dups (within class)
│       │   └── near_leak_*.csv         ← cross-split near-dups (within class)
│       ├── threshold_0.6/
│       ├── threshold_0.7/
│       ├── threshold_0.8/
│       └── threshold_0.9/
├── machine/
│   └── (same structure)
├── leakage_summary.json                ← all results in one file
└── README.md
```

---

## How to Interpret Results

### Exact duplicates
100% identical text appearing in two different splits. Most serious form of leakage.
Found once — independent of threshold.

### Near-duplicates
Texts sharing >= threshold% character n-gram overlap. Lower threshold = more pairs found.
Compare across thresholds to understand severity of similarity.

### If leakage is in HUMAN class only
Source corpus had duplicate documents. Machine texts (back-translated) naturally vary
even from the same source, so they don't duplicate exactly.

### If leakage is in MACHINE class only
The same human source document was used to generate machine samples in multiple splits.

### Near-duplicate threshold interpretation
If near-duplicate counts drop sharply as threshold increases (many pairs at 0.5, few at 0.8),
the similarity is loose and unlikely to have influenced the model. If counts stay high even
at 0.8–0.9, the similarity is tight and worth disclosing.

### Within-class vocabulary stability (`vocab_stability.csv`)
For each class, the top 100 words from train are tracked into eval and test.
A chi-squared test checks whether each word's usage rate shifted significantly.

| Result | Interpretation |
|---|---|
| High stability (> 80%) | Class vocabulary is consistent — splits were drawn from the same distribution |
| Low stability (< 50%) | Vocabulary drifts across splits — possible domain shift between train and eval/test |

Vocabulary instability does **not** indicate leakage — it indicates the opposite: the
evaluation data may come from a different distribution than training. This should be
disclosed and addressed in the limitations section.

---

## Citation
> Broder, A.Z. (1997). *On the resemblance and containment of documents.*
> Proceedings of the Compression and Complexity of Sequences (SEQUENCES '97), pp. 21-29. IEEE.
