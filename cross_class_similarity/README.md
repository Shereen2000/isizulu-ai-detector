# Cross-Class Similarity & Vocabulary Dominance — Results Guide

## What This Analysis Does

Two complementary analyses comparing **human vs machine** texts:

**1. Cross-class similarity** — for each split (train, eval, test), how many human and
machine texts are near-identical in character n-gram content? Runs across five thresholds.
This is not a problem — it is by design. The machine texts are back-translations of human
texts, so content overlap is expected. The question is: *how much overlap, and therefore
how hard was the classification task?*

**2. Vocabulary dominance** — which words appear disproportionately more in one class
than the other? Identifies the top 50 human-dominant and machine-dominant words in
train, then checks whether those same dominance patterns hold in eval and test.

---

## Methodology

| Parameter | Value |
|---|---|
| Thresholds tested | [0.5, 0.6, 0.7, 0.8, 0.9] |
| Shingling | Character 5-grams |
| MinHash permutations | 128 |
| Top N dominant words | 50 per class |
| Minimum word frequency | 10 occurrences in train |
| Significance test | Chi-squared (α = 0.05) |

---

## Folder Structure

```
cross_class_similarity/
├── README.md
├── cross_class_summary.json              ← similarity + vocab results in one file
├── splits/
│   ├── train/human.jsonl, machine.jsonl
│   ├── eval/
│   └── test/
├── results/
│   ├── threshold_0.5/
│   │   ├── train_cross_class_pairs.csv
│   │   ├── eval_cross_class_pairs.csv
│   │   └── test_cross_class_pairs.csv
│   ├── threshold_0.6/
│   ├── threshold_0.7/
│   ├── threshold_0.8/
│   └── threshold_0.9/
└── vocab_dominance/
    ├── train_vocab_dominance.csv         ← all words with dominance ratio + significance
    ├── eval_vocab_consistency.csv        ← do train-dominant words stay dominant in eval?
    ├── test_vocab_consistency.csv        ← do train-dominant words stay dominant in test?
    └── vocab_summary.json                ← aggregated counts and consistency percentages
```

---

## How to Interpret: Cross-Class Similarity

### Pair count
Number of human↔machine pairs within the same split sharing >= threshold character n-gram overlap.

| Count | Interpretation |
|---|---|
| Low (< 5%) | Classes are lexically distinct — task was easier |
| Moderate (5–15%) | Significant overlap — model had to learn subtle differences |
| High (> 15%) | Classes are very similar — strong evidence of generalisation |

### Why high similarity strengthens the argument
If human and machine texts share high n-gram overlap, the only available classification
signal is **linguistic texture**: fluency, naturalness, idiomatic expression, grammatical
patterns. High overlap means the model had no topic or vocabulary shortcut — it had to
learn what genuinely distinguishes human from machine writing.

---

## How to Interpret: Vocabulary Dominance

### Dominance ratio
How many times more frequently a word appears in its dominant class vs the other.
A ratio of 3.0x means the word appears 3× more often in that class.

### Cross-split consistency

| Result | Interpretation |
|---|---|
| High consistency (> 80%) | Dominant words are stable — model had vocabulary shortcuts |
| Low consistency (< 50%) | Dominant words flip between splits — no stable shortcut |

A model achieving 99%+ F1 even when vocabulary shortcuts are **inconsistent** across splits
has definitively learned writing style, not word frequency patterns.

---

## Citation
> Broder, A.Z. (1997). *On the resemblance and containment of documents.*
> Proceedings of the Compression and Complexity of Sequences (SEQUENCES '97), pp. 21–29. IEEE.
