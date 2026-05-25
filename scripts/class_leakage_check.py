import os
import json
import hashlib
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from datetime import datetime

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

SPLITS = {
    "train": os.path.join(PROJECT_DIR, "datasets", "train.jsonl"),
    "eval" : os.path.join(PROJECT_DIR, "datasets", "eval.jsonl"),
    "test" : os.path.join(PROJECT_DIR, "datasets", "test.jsonl"),
}

CLASSES = {0: "human", 1: "machine"}

THRESHOLD   = 0.7
NUM_PERM    = 128
NGRAM_SIZE  = 5
OUTPUT_ROOT = os.path.join(PROJECT_DIR, f"dataset_leakage_test > {THRESHOLD}")

print("=" * 70)
print("CLASS-LEVEL LEAKAGE DETECTION")
print(f"  Threshold : Jaccard >= {THRESHOLD}")
print("=" * 70)

# ── Helper functions ──────────────────────────────────────────────────────────
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

# ── Step 1: Split datasets by class and save ──────────────────────────────────
print("\nSTEP 1 — Splitting datasets by class")
print("─" * 70)

class_data = {0: {}, 1: {}}

for split_name, split_path in SPLITS.items():
    texts, labels = load_split(split_path)
    for class_id, class_name in CLASSES.items():
        class_texts  = [t for t, l in zip(texts, labels) if l == class_id]
        class_labels = [l for l in labels if l == class_id]

        # Save to folder
        out_dir  = os.path.join(OUTPUT_ROOT, class_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for text, label in zip(class_texts, class_labels):
                f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")

        class_data[class_id][split_name] = class_texts
        print(f"  {class_name:7s} / {split_name:5s}: {len(class_texts)} samples → {out_path}")

# ── Step 2: Leakage check per class ──────────────────────────────────────────
pairs = [("train", "eval"), ("train", "test"), ("eval", "test")]
all_summary = []

for class_id, class_name in CLASSES.items():
    print(f"\n{'='*70}")
    print(f"CLASS: {class_name.upper()}")
    print(f"{'='*70}")

    out_dir = os.path.join(OUTPUT_ROOT, class_name)
    class_summary = {"class": class_name, "pairs": []}

    for split_a, split_b in pairs:
        texts_a = class_data[class_id][split_a]
        texts_b = class_data[class_id][split_b]

        print(f"\n  {split_a} ↔ {split_b}  ({len(texts_a)} vs {len(texts_b)} samples)")

        # ── Exact duplicates ──────────────────────────────────────────────────
        hashes_a = {exact_hash(t): i for i, t in enumerate(texts_a)}
        exact_leaks = []
        for j, text in enumerate(texts_b):
            h = exact_hash(text)
            if h in hashes_a:
                exact_leaks.append({
                    f"{split_a}_idx"  : hashes_a[h],
                    f"{split_b}_idx"  : j,
                    "text_snippet"    : text[:120],
                })
        print(f"    Exact duplicates  : {len(exact_leaks)}")

        if exact_leaks:
            pd.DataFrame(exact_leaks).to_csv(
                os.path.join(out_dir, f"exact_leak_{split_a}_{split_b}.csv"), index=False)

        # ── Near-duplicates (MinHash) ─────────────────────────────────────────
        lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
        for i, text in enumerate(texts_a):
            lsh.insert(f"{split_a}_{i}", make_minhash(text))

        near_leaks = []
        for j, text in enumerate(texts_b):
            m = make_minhash(text)
            for match_key in lsh.query(m):
                i = int(match_key.split("_")[1])
                jaccard = m.jaccard(make_minhash(texts_a[i]))
                near_leaks.append({
                    f"{split_a}_idx"    : i,
                    f"{split_b}_idx"    : j,
                    "jaccard_similarity": round(jaccard, 4),
                    f"{split_a}_snippet": texts_a[i][:100],
                    f"{split_b}_snippet": texts_b[j][:100],
                })

        print(f"    Near-duplicates   : {len(near_leaks)}  (Jaccard >= {THRESHOLD})")

        if near_leaks:
            df = pd.DataFrame(near_leaks).sort_values("jaccard_similarity", ascending=False)
            df.to_csv(os.path.join(out_dir, f"near_leak_{split_a}_{split_b}.csv"), index=False)
            print(f"    Top 3 most similar:")
            for _, row in df.head(3).iterrows():
                print(f"      Jaccard={row['jaccard_similarity']:.3f}  "
                      f"{split_a}: {row[f'{split_a}_snippet'][:60]}...")

        # Percentage is relative to the split being queried (split_b = eval or test)
        base = len(texts_b)
        exact_pct  = round(100 * len(exact_leaks) / base, 2) if base else 0
        near_pct   = round(100 * len(near_leaks)  / base, 2) if base else 0

        class_summary["pairs"].append({
            "pair"                : f"{split_a}↔{split_b}",
            f"{split_b}_size"     : base,
            "exact_leaks"         : len(exact_leaks),
            "exact_leaks_pct"     : f"{exact_pct}%",
            "near_duplicates"     : len(near_leaks),
            "near_duplicates_pct" : f"{near_pct}%",
        })

    all_summary.append(class_summary)

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")

for cs in all_summary:
    print(f"\n  Class: {cs['class'].upper()}")
    print(f"  {'Pair':<15} {'Size':>6} {'Exact':>7} {'Exact%':>8} {'Near-dup(>=' + str(int(THRESHOLD*100)) + '%)':>14} {'Near%':>7}")
    print(f"  {'─'*63}")
    for p in cs["pairs"]:
        pair_parts = p['pair'].split('↔')
        size_key = f"{pair_parts[1]}_size"
        print(f"  {p['pair']:<15} {p.get(size_key, '?'):>6} "
              f"{p['exact_leaks']:>7} {p['exact_leaks_pct']:>8} "
              f"{p['near_duplicates']:>14} {p['near_duplicates_pct']:>7}")

summary_path = os.path.join(OUTPUT_ROOT, "class_leakage_summary.json")
with open(summary_path, "w") as f:
    json.dump({
        "near_duplicate_threshold": f">= {int(THRESHOLD*100)}% Jaccard similarity",
        "ngram_size"              : NGRAM_SIZE,
        "minhash_permutations"    : NUM_PERM,
        "results"                 : all_summary,
    }, f, indent=2)

print(f"\nAll outputs saved to: {OUTPUT_ROOT}")
print(f"  dataset_leakage_test/human/   — human class splits + leakage CSVs")
print(f"  dataset_leakage_test/machine/ — machine class splits + leakage CSVs")
print(f"  class_leakage_summary.json    — full summary")

# ── Write interpretation guide ────────────────────────────────────────────────
readme_path = os.path.join(OUTPUT_ROOT, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(f"""# Class-Level Leakage Detection — Results Guide

## What This Analysis Does

This audit checks whether the same (or near-identical) text appears in more than one
dataset split (train, eval, test), analysed **independently per class** (human vs machine).

Splitting by class before running leakage detection answers a more precise question than
a combined check: *within each class, did the same source documents bleed across splits?*

---

## Methodology

| Parameter | Value |
|---|---|
| Near-duplicate threshold | Jaccard similarity >= {int(THRESHOLD*100)}% |
| Shingling | Character {NGRAM_SIZE}-grams |
| MinHash permutations | {NUM_PERM} |
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
| `human/near_leak_*.csv` | Near-duplicate pairs (>= {int(THRESHOLD*100)}% similar) within the human class |
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

### Near-Duplicates (>= {int(THRESHOLD*100)}% Jaccard)

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
""")

print(f"  README.md                     — interpretation guide")
print(f"\nDone at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
