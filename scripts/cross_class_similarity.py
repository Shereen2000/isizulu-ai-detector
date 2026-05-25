import os
import json
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

THRESHOLD  = 0.7
NUM_PERM   = 128
NGRAM_SIZE = 5
OUTPUT_DIR = os.path.join(PROJECT_DIR, "cross_class_similarity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CROSS-CLASS SIMILARITY — HUMAN vs MACHINE WITHIN EACH SPLIT")
print(f"  Threshold  : Jaccard >= {THRESHOLD}")
print(f"  N-grams    : character {NGRAM_SIZE}-grams")
print(f"  MinHash    : {NUM_PERM} permutations")
print(f"  Error margin: ~{round(100 * (0.7*0.3/NUM_PERM)**0.5, 1)}% at threshold {THRESHOLD}")
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

def make_minhash(text):
    m = MinHash(num_perm=NUM_PERM)
    text = text.lower()
    for i in range(len(text) - NGRAM_SIZE + 1):
        m.update(text[i:i+NGRAM_SIZE].encode("utf-8"))
    return m

# ── Run per split ─────────────────────────────────────────────────────────────
all_summary = []

for split_name, split_path in SPLITS.items():
    print(f"\n{'='*70}")
    print(f"SPLIT: {split_name.upper()}")
    print(f"{'='*70}")

    texts, labels = load_split(split_path)
    labels = np.array(labels)

    human_texts   = [t for t, l in zip(texts, labels) if l == 0]
    machine_texts = [t for t, l in zip(texts, labels) if l == 1]

    n_human   = len(human_texts)
    n_machine = len(machine_texts)
    print(f"  Human   samples : {n_human}")
    print(f"  Machine samples : {n_machine}")

    # Build MinHash index on human texts
    print(f"\n  Building MinHash index on human texts...")
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    for i, text in enumerate(human_texts):
        lsh.insert(f"human_{i}", make_minhash(text))

    # Query with machine texts
    print(f"  Querying with machine texts...")
    pairs = []
    for j, text in enumerate(machine_texts):
        m = make_minhash(text)
        for match_key in lsh.query(m):
            i = int(match_key.split("_")[1])
            jaccard = m.jaccard(make_minhash(human_texts[i]))
            pairs.append({
                "human_idx"         : i,
                "machine_idx"       : j,
                "jaccard_similarity": round(jaccard, 4),
                "human_snippet"     : human_texts[i][:120],
                "machine_snippet"   : machine_texts[j][:120],
            })

    n_pairs = len(pairs)
    pct_human   = round(100 * n_pairs / n_human,   2)
    pct_machine = round(100 * n_pairs / n_machine, 2)

    print(f"\n  Cross-class pairs found : {n_pairs}")
    print(f"  As % of human   texts  : {pct_human}%")
    print(f"  As % of machine texts  : {pct_machine}%")

    if pairs:
        df = pd.DataFrame(pairs).sort_values("jaccard_similarity", ascending=False)

        # Jaccard distribution buckets
        buckets = {
            "0.70–0.79": ((df["jaccard_similarity"] >= 0.70) & (df["jaccard_similarity"] < 0.80)).sum(),
            "0.80–0.89": ((df["jaccard_similarity"] >= 0.80) & (df["jaccard_similarity"] < 0.90)).sum(),
            "0.90–0.99": ((df["jaccard_similarity"] >= 0.90) & (df["jaccard_similarity"] < 1.00)).sum(),
            "1.00"     : (df["jaccard_similarity"] == 1.00).sum(),
        }

        print(f"\n  Jaccard distribution:")
        for bucket, count in buckets.items():
            bar = "█" * min(count, 40)
            print(f"    {bucket} : {count:5d}  {bar}")

        print(f"\n  Top 5 most similar human↔machine pairs:")
        for _, row in df.head(5).iterrows():
            print(f"    Jaccard={row['jaccard_similarity']:.3f}")
            print(f"      Human  : {row['human_snippet'][:80]}...")
            print(f"      Machine: {row['machine_snippet'][:80]}...")

        df.to_csv(os.path.join(OUTPUT_DIR, f"{split_name}_cross_class_pairs.csv"), index=False)

    all_summary.append({
        "split"              : split_name,
        "human_samples"      : n_human,
        "machine_samples"    : n_machine,
        "cross_class_pairs"  : n_pairs,
        "pct_of_human"       : f"{pct_human}%",
        "pct_of_machine"     : f"{pct_machine}%",
        "jaccard_distribution": buckets if pairs else {},
    })

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")
print(f"\n  {'Split':<8} {'Human':>7} {'Machine':>9} {'Pairs':>7} {'% Human':>9} {'% Machine':>11}")
print(f"  {'─'*55}")
for s in all_summary:
    print(f"  {s['split']:<8} {s['human_samples']:>7} {s['machine_samples']:>9} "
          f"{s['cross_class_pairs']:>7} {s['pct_of_human']:>9} {s['pct_of_machine']:>11}")

# Save summary
summary_path = os.path.join(OUTPUT_DIR, "cross_class_summary.json")
with open(summary_path, "w") as f:
    json.dump({
        "description"           : "Within-split cross-class similarity: human vs machine texts in the same split",
        "threshold"             : f">= {int(THRESHOLD*100)}% Jaccard similarity",
        "ngram_size"            : NGRAM_SIZE,
        "minhash_permutations"  : NUM_PERM,
        "error_margin"          : f"~{round(100 * (0.7*0.3/NUM_PERM)**0.5, 1)}% at threshold {THRESHOLD}",
        "interpretation"        : (
            "High cross-class similarity within a split means the model could not rely on "
            "domain or topic to distinguish classes — it had to learn genuine linguistic "
            "differences between human and machine writing. The higher this number, "
            "the stronger the generalisation argument for model performance."
        ),
        "results": all_summary,
    }, f, indent=2)

# Write README
readme_path = os.path.join(OUTPUT_DIR, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(f"""# Cross-Class Similarity — Results Guide

## What This Analysis Does

For each split (train, eval, test), this script checks how many **human texts** and
**machine texts** within the same split are near-identical to each other (Jaccard >= {int(THRESHOLD*100)}%).

This is different from leakage detection. This is **not** a problem — it is by design.
The machine texts were produced by translating human texts to English and back to isiZulu.
So a human text and its machine counterpart are expected to be topically similar.

The question this answers is: **how similar are they?** And therefore: **how hard was
the classification task?**

---

## Methodology

| Parameter | Value |
|---|---|
| Similarity threshold | Jaccard >= {int(THRESHOLD*100)}% |
| Shingling | Character {NGRAM_SIZE}-grams |
| MinHash permutations | {NUM_PERM} |
| Error margin | ~{round(100 * (0.7*0.3/NUM_PERM)**0.5, 1)}% at threshold {THRESHOLD} (Broder, 1997) |

---

## How to Interpret the Results

### Cross-class pairs count
The number of human↔machine pairs within the same split that share >= {int(THRESHOLD*100)}% character n-gram overlap.

| Count | Interpretation |
|---|---|
| Low (< 5%) | Classes are lexically distinct — task was easier |
| Moderate (5–15%) | Significant overlap — model had to learn subtle differences |
| High (> 15%) | Classes are very similar — strong evidence of generalisation if model still performs well |

### Jaccard distribution
Shows whether the similarity is loose (0.70–0.79) or very tight (0.90–1.00).
A concentration at 0.90+ means human and machine texts are nearly identical in content,
making the classification task harder and the model's performance more impressive.

### Why high similarity strengthens the research argument
If human and machine texts in the same split are highly similar, the model could not have
relied on topic, domain, or vocabulary as classification signals — both classes share them.
The only signal available was the **linguistic texture** of the writing: naturalness,
fluency, idiomatic expression, grammatical patterns. This is exactly what an AI detector
should learn, and high cross-class similarity is the evidence that it had no other choice.

---

## Output Files

| File | Contents |
|---|---|
| `train_cross_class_pairs.csv` | All human↔machine similar pairs found in train |
| `eval_cross_class_pairs.csv` | All human↔machine similar pairs found in eval |
| `test_cross_class_pairs.csv` | All human↔machine similar pairs found in test |
| `cross_class_summary.json` | Aggregated counts, percentages, and Jaccard distribution |

### CSV columns
| Column | Meaning |
|---|---|
| `human_idx` | Index of the human text within the human subset of the split |
| `machine_idx` | Index of the machine text within the machine subset of the split |
| `jaccard_similarity` | Similarity score (0.0 – 1.0) |
| `human_snippet` | First 120 characters of the human text |
| `machine_snippet` | First 120 characters of the machine text |

---

## Citation
> Broder, A.Z. (1997). *On the resemblance and containment of documents.*
> Proceedings of the Compression and Complexity of Sequences (SEQUENCES '97), pp. 21–29. IEEE.
""")

print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  <split>_cross_class_pairs.csv — all similar human↔machine pairs per split")
print(f"  cross_class_summary.json      — aggregated results")
print(f"  README.md                     — interpretation guide")
print(f"\nDone at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
