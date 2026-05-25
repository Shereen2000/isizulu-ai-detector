import os
import json
import hashlib
import re
import numpy as np
import pandas as pd
from collections import Counter
from datasketch import MinHash, MinHashLSH
from scipy.stats import chi2_contingency
from datetime import datetime

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

SPLITS = {
    "train": os.path.join(PROJECT_DIR, "datasets", "train.jsonl"),
    "eval" : os.path.join(PROJECT_DIR, "datasets", "eval.jsonl"),
    "test" : os.path.join(PROJECT_DIR, "datasets", "test.jsonl"),
}

CLASSES        = {0: "human", 1: "machine"}
THRESHOLDS     = [0.5, 0.6, 0.7, 0.8, 0.9]
NUM_PERM       = 128
NGRAM_SIZE     = 5
TOP_WORDS      = 30
MIN_VOCAB_FREQ = 5
CHI2_ALPHA     = 0.05
OUTPUT_ROOT    = os.path.join(PROJECT_DIR, "dataset_leakage_test")

print("=" * 70)
print("CLASS-LEVEL LEAKAGE & VOCABULARY STABILITY — ALL THRESHOLDS")
print(f"  Thresholds  : {THRESHOLDS}")
print(f"  N-grams     : character {NGRAM_SIZE}-grams")
print(f"  MinHash     : {NUM_PERM} permutations")
print(f"  Top words   : {TOP_WORDS} per class (min freq={MIN_VOCAB_FREQ})")
print("=" * 70)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_split(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            texts.append(s["text"])
            labels.append(int(s["label"]))
    return texts, labels

def exact_hash(text):
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

def make_minhash(text):
    m = MinHash(num_perm=NUM_PERM)
    text = text.lower()
    for i in range(len(text) - NGRAM_SIZE + 1):
        m.update(text[i:i+NGRAM_SIZE].encode("utf-8"))
    return m

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def word_frequencies(texts):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    return counter

# ── Step 1: Split datasets by class and save once ─────────────────────────────
print("\nSTEP 1 — Splitting datasets by class and saving")
print("─" * 70)

class_data = {0: {}, 1: {}}

for split_name, split_path in SPLITS.items():
    texts, labels = load_split(split_path)
    for class_id, class_name in CLASSES.items():
        class_texts  = [t for t, l in zip(texts, labels) if l == class_id]
        class_labels = [l for l in labels if l == class_id]

        out_dir  = os.path.join(OUTPUT_ROOT, class_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for text, label in zip(class_texts, class_labels):
                f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")

        class_data[class_id][split_name] = class_texts
        print(f"  {class_name:7s} / {split_name:5s}: {len(class_texts)} samples → {out_path}")

# ── Step 2: Pre-compute MinHashes for all texts ───────────────────────────────
print(f"\nSTEP 2 — Pre-computing MinHashes")
print("─" * 70)

minhashes = {0: {}, 1: {}}
for class_id in CLASSES:
    for split_name in SPLITS:
        texts = class_data[class_id][split_name]
        minhashes[class_id][split_name] = [make_minhash(t) for t in texts]
        print(f"  {CLASSES[class_id]:7s} / {split_name:5s}: {len(texts)} hashes computed")

# ── Step 3: Exact duplicates (threshold-independent) ─────────────────────────
print(f"\nSTEP 3 — Exact duplicate check (threshold-independent)")
print("─" * 70)

split_pairs = [("train", "eval"), ("train", "test"), ("eval", "test")]
exact_results = {CLASSES[cid]: {} for cid in CLASSES}

for class_id, class_name in CLASSES.items():
    out_dir = os.path.join(OUTPUT_ROOT, class_name)
    for split_a, split_b in split_pairs:
        texts_a = class_data[class_id][split_a]
        texts_b = class_data[class_id][split_b]
        base    = len(texts_b)

        hashes_a = {exact_hash(t): i for i, t in enumerate(texts_a)}
        exact_leaks = []
        for j, text in enumerate(texts_b):
            h = exact_hash(text)
            if h in hashes_a:
                exact_leaks.append({
                    f"{split_a}_idx": hashes_a[h],
                    f"{split_b}_idx": j,
                    "text_snippet"  : text[:120],
                })

        pct = round(100 * len(exact_leaks) / base, 2) if base else 0
        print(f"  {class_name:7s} {split_a}↔{split_b}: {len(exact_leaks)} exact leaks ({pct}% of {split_b})")

        if exact_leaks:
            pd.DataFrame(exact_leaks).to_csv(
                os.path.join(out_dir, f"exact_leak_{split_a}_{split_b}.csv"), index=False)

        exact_results[class_name][f"{split_a}↔{split_b}"] = {
            f"{split_b}_size" : base,
            "exact_leaks"     : len(exact_leaks),
            "exact_leaks_pct" : f"{pct}%",
        }

# ── Step 4: Within-class vocabulary stability across splits ───────────────────
print(f"\nSTEP 4 — Within-class vocabulary stability across splits")
print("─" * 70)

vocab_stability_results = {}

for class_id, class_name in CLASSES.items():
    out_dir     = os.path.join(OUTPUT_ROOT, class_name)
    train_texts = class_data[class_id]["train"]
    train_freq  = word_frequencies(train_texts)
    n_train     = len(train_texts)

    top_words = [w for w, c in train_freq.most_common(TOP_WORDS * 3) if c >= MIN_VOCAB_FREQ][:TOP_WORDS]

    print(f"\n  Class: {class_name.upper()} — tracking top {len(top_words)} words from train")

    rows = []
    for split_name in ["eval", "test"]:
        split_texts = class_data[class_id][split_name]
        split_freq  = word_frequencies(split_texts)
        n_split     = len(split_texts)

        for word in top_words:
            tc = train_freq.get(word, 0)
            sc = split_freq.get(word, 0)

            table = [[tc, n_train - tc], [sc, n_split - sc]]
            try:
                chi2, p, _, _ = chi2_contingency(table)
            except Exception:
                chi2, p = 0, 1.0

            train_rate = round(tc / n_train, 6) if n_train else 0
            split_rate = round(sc / n_split, 6) if n_split else 0

            rows.append({
                "word"              : word,
                "compared_split"    : split_name,
                "train_rate"        : train_rate,
                f"{split_name}_rate": split_rate,
                "rate_diff"         : round(abs(train_rate - split_rate), 6),
                "chi2"              : round(chi2, 4),
                "p_value"           : round(p, 6),
                "significant_shift" : p < CHI2_ALPHA,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "vocab_stability.csv"), index=False)

    class_vocab = {"top_words_tracked": len(top_words)}
    for split_name in ["eval", "test"]:
        sub      = df[df["compared_split"] == split_name]
        shifted  = int(sub["significant_shift"].sum())
        stable   = len(top_words) - shifted
        pct_stab = round(100 * stable / len(top_words), 1) if top_words else 0
        class_vocab[split_name] = {
            "stable"    : stable,
            "shifted"   : shifted,
            "pct_stable": f"{pct_stab}%",
        }
        print(f"    train↔{split_name}: {shifted}/{len(top_words)} words shifted significantly "
              f"({pct_stab}% stable)")

    vocab_stability_results[class_name] = class_vocab

# ── Step 5: Near-duplicate check per threshold ────────────────────────────────
print(f"\nSTEP 5 — Near-duplicate check across all thresholds")
print("─" * 70)

all_threshold_results = []

for threshold in THRESHOLDS:
    print(f"\n  Threshold: {threshold}")
    threshold_summary = {"threshold": threshold, "classes": []}

    for class_id, class_name in CLASSES.items():
        out_dir = os.path.join(OUTPUT_ROOT, class_name, "results", f"threshold_{threshold}")
        os.makedirs(out_dir, exist_ok=True)
        class_summary = {"class": class_name, "pairs": []}

        for split_a, split_b in split_pairs:
            texts_a  = class_data[class_id][split_a]
            texts_b  = class_data[class_id][split_b]
            hashes_a = minhashes[class_id][split_a]
            hashes_b = minhashes[class_id][split_b]
            base     = len(texts_b)

            lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
            for i, m in enumerate(hashes_a):
                lsh.insert(f"{split_a}_{i}", m)

            near_leaks = []
            for j, m in enumerate(hashes_b):
                for match_key in lsh.query(m):
                    i       = int(match_key.split("_")[1])
                    jaccard = round(m.jaccard(hashes_a[i]), 4)
                    if jaccard < threshold:
                        continue
                    near_leaks.append({
                        f"{split_a}_idx"    : i,
                        f"{split_b}_idx"    : j,
                        "jaccard_similarity": jaccard,
                        f"{split_a}_snippet": texts_a[i][:100],
                        f"{split_b}_snippet": texts_b[j][:100],
                    })

            pct = round(100 * len(near_leaks) / base, 2) if base else 0
            print(f"    {class_name:7s} {split_a}↔{split_b}: {len(near_leaks)} near-dups ({pct}% of {split_b})")

            if near_leaks:
                df = pd.DataFrame(near_leaks).sort_values("jaccard_similarity", ascending=False)
                df.to_csv(os.path.join(out_dir, f"near_leak_{split_a}_{split_b}.csv"), index=False)

            class_summary["pairs"].append({
                "pair"               : f"{split_a}↔{split_b}",
                f"{split_b}_size"    : base,
                "near_duplicates"    : len(near_leaks),
                "near_duplicates_pct": f"{pct}%",
            })

        threshold_summary["classes"].append(class_summary)

    all_threshold_results.append(threshold_summary)

# ── Save summary ──────────────────────────────────────────────────────────────
summary = {
    "ngram_size"              : NGRAM_SIZE,
    "minhash_permutations"    : NUM_PERM,
    "thresholds_tested"       : THRESHOLDS,
    "vocab_top_words"         : TOP_WORDS,
    "vocab_min_freq"          : MIN_VOCAB_FREQ,
    "exact_results"           : exact_results,
    "vocab_stability_results" : vocab_stability_results,
    "near_duplicate_results"  : all_threshold_results,
}

with open(os.path.join(OUTPUT_ROOT, "leakage_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ── Final summary tables ──────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("FINAL SUMMARY — NEAR-DUPLICATES BY THRESHOLD")
print(f"{'='*70}")

for class_name in ["human", "machine"]:
    print(f"\n  Class: {class_name.upper()}")
    print(f"  {'Pair':<15} {'Size':>6} " + " ".join(f"  {t:.1f}%" for t in THRESHOLDS))
    print(f"  {'─'*70}")
    for pair in [f"{a}↔{b}" for a, b in split_pairs]:
        size = None
        counts = []
        for tr in all_threshold_results:
            for cs in tr["classes"]:
                if cs["class"] == class_name:
                    for p in cs["pairs"]:
                        if p["pair"] == pair:
                            counts.append(p["near_duplicates_pct"])
                            if size is None:
                                split_b = pair.split("↔")[1]
                                size = p.get(f"{split_b}_size", "?")
        print(f"  {pair:<15} {str(size):>6} " + " ".join(f"{c:>6}" for c in counts))

print(f"\n{'='*70}")
print("FINAL SUMMARY — WITHIN-CLASS VOCABULARY STABILITY")
print(f"{'='*70}")
for class_name in ["human", "machine"]:
    r = vocab_stability_results[class_name]
    print(f"\n  Class: {class_name.upper()} (top {r['top_words_tracked']} words tracked)")
    for split_name in ["eval", "test"]:
        sr = r[split_name]
        print(f"    train↔{split_name}: {sr['stable']}/{r['top_words_tracked']} stable "
              f"({sr['pct_stable']}), {sr['shifted']} shifted")

# ── Write README ──────────────────────────────────────────────────────────────
with open(os.path.join(OUTPUT_ROOT, "README.md"), "w", encoding="utf-8") as f:
    f.write(f"""# Class-Level Leakage & Vocabulary Stability — Results Guide

## What This Analysis Does

Two complementary checks, both run independently per class (human vs machine):

**1. Leakage detection** — checks whether the same (or near-identical) text appears
across splits (train, eval, test). Exact duplicates and near-duplicates at five thresholds.

**2. Within-class vocabulary stability** — checks whether the top {TOP_WORDS} most frequent
words in each class's training split maintain similar rates in eval and test. If they do,
the class vocabulary is stable across splits. If they shift, there is domain drift.

---

## Methodology

| Parameter | Value |
|---|---|
| Thresholds tested | {THRESHOLDS} |
| Shingling | Character {NGRAM_SIZE}-grams |
| MinHash permutations | {NUM_PERM} |
| Exact duplicate method | MD5 hash of lowercased, stripped text |
| Top words tracked (vocab) | {TOP_WORDS} per class |
| Min word frequency (vocab) | {MIN_VOCAB_FREQ} occurrences in train |
| Significance test (vocab) | Chi-squared (α = {CHI2_ALPHA}) |

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
│       │   └── near_leak_*.csv
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
For each class, the top {TOP_WORDS} words from train are tracked into eval and test.
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
""")

print(f"\nOutputs saved to: {OUTPUT_ROOT}")
print(f"  human/ machine/          — class-split jsonl files + exact leak CSVs")
print(f"  */vocab_stability.csv    — within-class word rate stability")
print(f"  */results/threshold_*/   — near-duplicate CSVs per threshold")
print(f"  leakage_summary.json     — all results in one file")
print(f"  README.md                — interpretation guide")
print(f"\nDone at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
