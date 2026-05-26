import os
import json
import re
import numpy as np
import pandas as pd
from collections import Counter
from datasketch import MinHash, MinHashLSH
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

SPLITS = {
    "train": os.path.join(PROJECT_DIR, "datasets", "train.jsonl"),
    "eval" : os.path.join(PROJECT_DIR, "datasets", "eval.jsonl"),
    "test" : os.path.join(PROJECT_DIR, "datasets", "test.jsonl"),
}

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
NUM_PERM   = 128
NGRAM_SIZE = 5
TOP_N      = 100
MIN_FREQ   = 5
CHI2_ALPHA = 0.05
OUTPUT_DIR = os.path.join(PROJECT_DIR, "cross_class_similarity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CROSS-CLASS SIMILARITY & VOCABULARY DOMINANCE")
print(f"  Thresholds : {THRESHOLDS}")
print(f"  N-grams    : character {NGRAM_SIZE}-grams")
print(f"  MinHash    : {NUM_PERM} permutations")
print(f"  Top words  : {TOP_N} per class (min freq={MIN_FREQ})")
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

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def word_frequencies(texts):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    return counter

def dominance_ratio(count_a, total_a, count_b, total_b):
    rate_a = count_a / total_a if total_a else 0
    rate_b = count_b / total_b if total_b else 0
    if rate_b == 0:
        return float('inf')
    return round(rate_a / rate_b, 4)

# ── Step 1: Load splits, save class-split files, pre-compute MinHashes ────────
print("\nSTEP 1 — Loading splits and pre-computing MinHashes")
print("─" * 70)

split_data   = {}
split_hashes = {}

for split_name, split_path in SPLITS.items():
    texts, labels = load_split(split_path)
    labels = np.array(labels)

    human_texts   = [t for t, l in zip(texts, labels) if l == 0]
    machine_texts = [t for t, l in zip(texts, labels) if l == 1]

    split_data[split_name] = {"human": human_texts, "machine": machine_texts}

    split_dir = os.path.join(OUTPUT_DIR, "splits", split_name)
    os.makedirs(split_dir, exist_ok=True)
    for class_name, class_texts, class_label in [
        ("human", human_texts, 0), ("machine", machine_texts, 1)
    ]:
        out_path = os.path.join(split_dir, f"{class_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for text in class_texts:
                f.write(json.dumps({"text": text, "label": class_label}, ensure_ascii=False) + "\n")

    print(f"  {split_name:5s}: {len(human_texts)} human, {len(machine_texts)} machine — splits saved")

    human_hashes   = [make_minhash(t) for t in human_texts]
    machine_hashes = [make_minhash(t) for t in machine_texts]
    split_hashes[split_name] = {"human": human_hashes, "machine": machine_hashes}
    print(f"         {len(human_hashes)} human hashes + {len(machine_hashes)} machine hashes computed")

# ── Step 2: Cross-class similarity for each threshold ─────────────────────────
print(f"\nSTEP 2 — Cross-class similarity across all thresholds")
print("─" * 70)

all_threshold_results = []

for threshold in THRESHOLDS:
    print(f"\n  Threshold: {threshold}")
    threshold_summary = {"threshold": threshold, "splits": []}

    for split_name in SPLITS:
        human_texts    = split_data[split_name]["human"]
        machine_texts  = split_data[split_name]["machine"]
        human_hashes   = split_hashes[split_name]["human"]
        machine_hashes = split_hashes[split_name]["machine"]

        n_human   = len(human_texts)
        n_machine = len(machine_texts)

        lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
        for i, m in enumerate(human_hashes):
            lsh.insert(f"human_{i}", m)

        pairs = []
        for j, m in enumerate(machine_hashes):
            for match_key in lsh.query(m):
                i       = int(match_key.split("_")[1])
                jaccard = round(m.jaccard(human_hashes[i]), 4)
                if jaccard < threshold:
                    continue
                pairs.append({
                    "human_idx"         : i,
                    "machine_idx"       : j,
                    "jaccard_similarity": jaccard,
                    "human_snippet"     : human_texts[i][:120],
                    "machine_snippet"   : machine_texts[j][:120],
                })

        n_pairs     = len(pairs)
        pct_human   = round(100 * n_pairs / n_human,   2) if n_human   else 0
        pct_machine = round(100 * n_pairs / n_machine, 2) if n_machine else 0

        print(f"    {split_name:5s}: {n_pairs} pairs  ({pct_human}% of human, {pct_machine}% of machine)")

        if pairs:
            out_dir = os.path.join(OUTPUT_DIR, "results", f"threshold_{threshold}")
            os.makedirs(out_dir, exist_ok=True)
            df = pd.DataFrame(pairs).sort_values("jaccard_similarity", ascending=False)
            df.to_csv(os.path.join(out_dir, f"{split_name}_cross_class_pairs.csv"), index=False)

            buckets = {
                "0.50–0.59": int(((df["jaccard_similarity"] >= 0.50) & (df["jaccard_similarity"] < 0.60)).sum()),
                "0.60–0.69": int(((df["jaccard_similarity"] >= 0.60) & (df["jaccard_similarity"] < 0.70)).sum()),
                "0.70–0.79": int(((df["jaccard_similarity"] >= 0.70) & (df["jaccard_similarity"] < 0.80)).sum()),
                "0.80–0.89": int(((df["jaccard_similarity"] >= 0.80) & (df["jaccard_similarity"] < 0.90)).sum()),
                "0.90–0.99": int(((df["jaccard_similarity"] >= 0.90) & (df["jaccard_similarity"] < 1.00)).sum()),
                "1.00"     : int((df["jaccard_similarity"] == 1.00).sum()),
            }
        else:
            buckets = {}

        threshold_summary["splits"].append({
            "split"              : split_name,
            "human_samples"      : n_human,
            "machine_samples"    : n_machine,
            "cross_class_pairs"  : n_pairs,
            "pct_of_human"       : f"{pct_human}%",
            "pct_of_machine"     : f"{pct_machine}%",
            "jaccard_distribution": buckets,
        })

    all_threshold_results.append(threshold_summary)

# ── Step 3: Cross-class vocabulary dominance ──────────────────────────────────
print(f"\nSTEP 3 — Cross-class vocabulary dominance")
print("─" * 70)

train_human   = split_data["train"]["human"]
train_machine = split_data["train"]["machine"]
n_human_train   = len(train_human)
n_machine_train = len(train_machine)

human_freq   = word_frequencies(train_human)
machine_freq = word_frequencies(train_machine)

print(f"  Train — human: {n_human_train} texts, machine: {n_machine_train} texts")

all_words = (
    set(w for w, c in human_freq.items()   if c >= MIN_FREQ) |
    set(w for w, c in machine_freq.items() if c >= MIN_FREQ)
)
print(f"  Vocabulary size (min freq={MIN_FREQ}): {len(all_words)} words")

rows = []
for word in all_words:
    hc = human_freq.get(word, 0)
    mc = machine_freq.get(word, 0)

    table = [[hc, n_human_train - hc], [mc, n_machine_train - mc]]
    try:
        chi2, p, _, _ = chi2_contingency(table)
    except Exception:
        chi2, p = 0, 1.0

    human_rate   = round(hc / n_human_train,   6)
    machine_rate = round(mc / n_machine_train,  6)

    if human_rate >= machine_rate:
        dominant_class = "human"
        ratio = dominance_ratio(hc, n_human_train, mc, n_machine_train)
    else:
        dominant_class = "machine"
        ratio = dominance_ratio(mc, n_machine_train, hc, n_human_train)

    rows.append({
        "word"           : word,
        "human_count"    : hc,
        "machine_count"  : mc,
        "human_rate"     : human_rate,
        "machine_rate"   : machine_rate,
        "dominant_class" : dominant_class,
        "dominance_ratio": ratio,
        "chi2"           : round(chi2, 4),
        "p_value"        : round(p, 6),
        "significant"    : p < CHI2_ALPHA,
    })

train_df    = pd.DataFrame(rows).sort_values("dominance_ratio", ascending=False)
top_human   = train_df[train_df["dominant_class"] == "human"].head(TOP_N)
top_machine = train_df[train_df["dominant_class"] == "machine"].head(TOP_N)

vocab_dir = os.path.join(OUTPUT_DIR, "vocab_dominance")
os.makedirs(vocab_dir, exist_ok=True)
train_df.to_csv(os.path.join(vocab_dir, "train_vocab_dominance.csv"), index=False)

print(f"\n  Top 10 human-dominant words in train:")
print(f"  {'Word':<20} {'Human rate':>12} {'Machine rate':>13} {'Ratio':>8} {'Sig':>5}")
print(f"  {'─'*60}")
for _, row in top_human.head(10).iterrows():
    print(f"  {row['word']:<20} {row['human_rate']:>12.4f} {row['machine_rate']:>13.4f} "
          f"{row['dominance_ratio']:>8.2f}x {'Y' if row['significant'] else 'n':>5}")

print(f"\n  Top 10 machine-dominant words in train:")
print(f"  {'Word':<20} {'Machine rate':>12} {'Human rate':>13} {'Ratio':>8} {'Sig':>5}")
print(f"  {'─'*60}")
for _, row in top_machine.head(10).iterrows():
    print(f"  {row['word']:<20} {row['machine_rate']:>12.4f} {row['human_rate']:>13.4f} "
          f"{row['dominance_ratio']:>8.2f}x {'Y' if row['significant'] else 'n':>5}")

# Track dominant words into eval and test
dominant_words  = set(top_human["word"].tolist() + top_machine["word"].tolist())
word_class_map  = dict(zip(train_df["word"], train_df["dominant_class"]))

print(f"\n  Tracking {len(dominant_words)} dominant train words into eval and test")
print(f"  {'─'*60}")

consistency_results = {}

for split_name in ["eval", "test"]:
    h_texts = split_data[split_name]["human"]
    m_texts = split_data[split_name]["machine"]
    n_h     = len(h_texts)
    n_m     = len(m_texts)
    h_freq  = word_frequencies(h_texts)
    m_freq  = word_frequencies(m_texts)

    rows = []
    for word in dominant_words:
        hc = h_freq.get(word, 0)
        mc = m_freq.get(word, 0)
        human_rate   = round(hc / n_h, 6) if n_h else 0
        machine_rate = round(mc / n_m, 6) if n_m else 0

        if human_rate > machine_rate:
            dominant_in_split = "human"
        elif machine_rate > human_rate:
            dominant_in_split = "machine"
        else:
            dominant_in_split = "equal"

        train_dominant = word_class_map.get(word, "unknown")
        rows.append({
            "word"                        : word,
            "train_dominant_class"        : train_dominant,
            "human_count"                 : hc,
            "machine_count"               : mc,
            "human_rate"                  : human_rate,
            "machine_rate"                : machine_rate,
            f"{split_name}_dominant_class": dominant_in_split,
            "consistent_with_train"       : dominant_in_split == train_dominant or dominant_in_split == "equal",
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(vocab_dir, f"{split_name}_vocab_consistency.csv"), index=False)

    n_consistent   = int(df["consistent_with_train"].sum())
    n_inconsistent = int((~df["consistent_with_train"]).sum())
    pct_consistent = round(100 * n_consistent / len(df), 2)

    print(f"\n  {split_name.upper()}: {n_consistent}/{len(df)} dominant words ({pct_consistent}%) "
          f"consistent with train dominance")

    if n_inconsistent > 0:
        flipped = df[~df["consistent_with_train"]].head(10)
        print(f"  Words that flipped:")
        for _, row in flipped.iterrows():
            print(f"    '{row['word']}': train={row['train_dominant_class']} → "
                  f"{split_name}={row[f'{split_name}_dominant_class']}")

    consistency_results[split_name] = {
        "total_tracked"  : int(len(df)),
        "consistent"     : n_consistent,
        "inconsistent"   : n_inconsistent,
        "pct_consistent" : f"{pct_consistent}%",
    }

sig_human   = int((top_human["significant"]   == True).sum())
sig_machine = int((top_machine["significant"] == True).sum())

vocab_summary = {
    "top_n_words_tracked"    : TOP_N,
    "min_frequency"          : MIN_FREQ,
    "chi2_significance_alpha": CHI2_ALPHA,
    "train_dominance": {
        "human_dominant_words"       : len(top_human),
        "human_dominant_significant" : sig_human,
        "machine_dominant_words"     : len(top_machine),
        "machine_dominant_significant": sig_machine,
    },
    "cross_split_consistency": consistency_results,
    "interpretation": (
        "If dominant words are consistent across splits, the model had a vocabulary "
        "shortcut available. If inconsistent, the model could not have relied on "
        "specific words to classify — it had to learn genuine writing style differences."
    ),
}

with open(os.path.join(vocab_dir, "vocab_summary.json"), "w") as f:
    json.dump(vocab_summary, f, indent=2)

# ── Step 4: Cross-class domain similarity (TF-IDF cosine) ────────────────────
print(f"\n{'─'*70}")
print("STEP 4 — Cross-class domain similarity (TF-IDF cosine)")
print(f"{'─'*70}")

all_train_texts = split_data["train"]["human"] + split_data["train"]["machine"]
vectorizer = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5),
    sublinear_tf=True
)
vectorizer.fit(all_train_texts)

domain_results = {}

for split_name in SPLITS:
    human_texts   = split_data[split_name]["human"]
    machine_texts = split_data[split_name]["machine"]

    X_human   = vectorizer.transform(human_texts)
    X_machine = vectorizer.transform(machine_texts)

    centroid_human   = np.asarray(X_human.mean(axis=0))
    centroid_machine = np.asarray(X_machine.mean(axis=0))

    cross_sim = round(float(
        cosine_similarity(centroid_human, centroid_machine)[0][0]), 4)

    sim_h = cosine_similarity(X_human)
    sim_m = cosine_similarity(X_machine)
    n_h, n_m = sim_h.shape[0], sim_m.shape[0]
    avg_human   = round(float(sim_h[np.triu_indices(n_h, k=1)].mean()), 4)
    avg_machine = round(float(sim_m[np.triu_indices(n_m, k=1)].mean()), 4)

    label = "same domain" if cross_sim >= 0.9 else "similar" if cross_sim >= 0.7 else "domain shift"
    print(f"\n  {split_name.upper()}: human↔machine cosine = {cross_sim:.4f}  ({label})")
    print(f"    Human diversity:   {avg_human:.4f}  ({n_h} texts, {n_h*(n_h-1)//2} pairs)  "
          f"({'low diversity' if avg_human >= 0.5 else 'moderate' if avg_human >= 0.3 else 'high diversity'})")
    print(f"    Machine diversity: {avg_machine:.4f}  ({n_m} texts, {n_m*(n_m-1)//2} pairs)  "
          f"({'low diversity' if avg_machine >= 0.5 else 'moderate' if avg_machine >= 0.3 else 'high diversity'})")

    domain_results[split_name] = {
        "cross_class_centroid_similarity": cross_sim,
        "intra_class_diversity": {
            "human"  : avg_human,
            "machine": avg_machine,
        },
    }

# ── Step 5: Text length distribution — human vs machine per split ─────────────
print(f"\n{'─'*70}")
print("STEP 5 — Text length: human vs machine per split")
print(f"{'─'*70}")

length_results   = {}
all_length_rows  = []

for split_name in SPLITS:
    human_texts   = split_data[split_name]["human"]
    machine_texts = split_data[split_name]["machine"]
    L_h = np.array([len(t) for t in human_texts])
    L_m = np.array([len(t) for t in machine_texts])

    stat, p = mannwhitneyu(L_h, L_m, alternative="two-sided")
    sig = p < CHI2_ALPHA

    print(f"\n  {split_name.upper()}:")
    print(f"    Human  : mean={L_h.mean():.0f}  median={np.median(L_h):.0f}  "
          f"std={L_h.std():.0f}  min={L_h.min()}  max={L_h.max()}")
    print(f"    Machine: mean={L_m.mean():.0f}  median={np.median(L_m):.0f}  "
          f"std={L_m.std():.0f}  min={L_m.min()}  max={L_m.max()}")
    print(f"    Human vs Machine Mann-Whitney U: p={p:.6f}  "
          f"{'SIGNIFICANT difference' if sig else 'no significant difference'}")

    for class_name, L in [("human", L_h), ("machine", L_m)]:
        all_length_rows.append({
            "split"  : split_name,
            "class"  : class_name,
            "n"      : len(L),
            "mean"   : round(float(L.mean()), 1),
            "median" : round(float(np.median(L)), 1),
            "std"    : round(float(L.std()), 1),
            "min"    : int(L.min()),
            "p25"    : round(float(np.percentile(L, 25)), 1),
            "p75"    : round(float(np.percentile(L, 75)), 1),
            "max"    : int(L.max()),
        })

    length_results[split_name] = {
        "human_mean"   : round(float(L_h.mean()), 1),
        "machine_mean" : round(float(L_m.mean()), 1),
        "human_median" : round(float(np.median(L_h)), 1),
        "machine_median": round(float(np.median(L_m)), 1),
        "mannwhitney_p": round(float(p), 6),
        "significant"  : bool(sig),
    }

pd.DataFrame(all_length_rows).to_csv(
    os.path.join(OUTPUT_DIR, "length_stats.csv"), index=False)

# ── Final summary tables ──────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("FINAL SUMMARY — CROSS-CLASS PAIRS BY THRESHOLD")
print(f"{'='*70}")

for split_name in SPLITS:
    print(f"\n  Split: {split_name.upper()}")
    print(f"  {'Threshold':<12} {'Pairs':>8} {'% Human':>10} {'% Machine':>12}")
    print(f"  {'─'*45}")
    for tr in all_threshold_results:
        for sp in tr["splits"]:
            if sp["split"] == split_name:
                print(f"  {tr['threshold']:<12} {sp['cross_class_pairs']:>8} "
                      f"{sp['pct_of_human']:>10} {sp['pct_of_machine']:>12}")

print(f"\n{'='*70}")
print("FINAL SUMMARY — VOCABULARY DOMINANCE CONSISTENCY")
print(f"{'='*70}")
print(f"\n  Train: {sig_human}/{len(top_human)} human-dominant significant, "
      f"{sig_machine}/{len(top_machine)} machine-dominant significant")
for split_name, res in consistency_results.items():
    print(f"  {split_name:5s}: {res['consistent']}/{res['total_tracked']} dominant words "
          f"consistent ({res['pct_consistent']})")

# ── Save overall summary JSON ─────────────────────────────────────────────────
summary = {
    "description"         : "Within-split cross-class similarity + vocabulary dominance",
    "ngram_size"          : NGRAM_SIZE,
    "minhash_permutations": NUM_PERM,
    "thresholds_tested"   : THRESHOLDS,
    "similarity_results"  : all_threshold_results,
    "vocab_dominance"     : vocab_summary,
    "domain_similarity"   : domain_results,
    "length_distribution" : length_results,
}

with open(os.path.join(OUTPUT_DIR, "cross_class_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ── Write README ──────────────────────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(f"""# Cross-Class Similarity & Vocabulary Dominance — Results Guide

## What This Analysis Does

Two complementary analyses comparing **human vs machine** texts:

**1. Cross-class similarity** — for each split (train, eval, test), how many human and
machine texts are near-identical in character n-gram content? Runs across five thresholds.
This is not a problem — it is by design. The machine texts are back-translations of human
texts, so content overlap is expected. The question is: *how much overlap, and therefore
how hard was the classification task?*

**2. Vocabulary dominance** — which words appear disproportionately more in one class
than the other? Identifies the top {TOP_N} human-dominant and machine-dominant words in
train, then checks whether those same dominance patterns hold in eval and test.

---

## Methodology

| Parameter | Value |
|---|---|
| Thresholds tested | {THRESHOLDS} |
| Shingling | Character {NGRAM_SIZE}-grams |
| MinHash permutations | {NUM_PERM} |
| Top N dominant words | {TOP_N} per class |
| Minimum word frequency | {MIN_FREQ} occurrences in train |
| Significance test | Chi-squared (α = {CHI2_ALPHA}) |

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
""")

print(f"\n{'='*70}")
print("FINAL SUMMARY — CROSS-CLASS DOMAIN SIMILARITY")
print(f"{'='*70}")
for split_name, res in domain_results.items():
    sim   = res["cross_class_centroid_similarity"]
    label = "same domain" if sim >= 0.9 else "similar" if sim >= 0.7 else "domain shift"
    print(f"  {split_name:5s}: human↔machine = {sim:.4f}  ({label})"
          f"  |  human div={res['intra_class_diversity']['human']:.4f}"
          f"  machine div={res['intra_class_diversity']['machine']:.4f}")

print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  splits/                      — class-split jsonl files per split")
print(f"  results/threshold_*/         — cross-class pair CSVs per threshold")
print(f"  vocab_dominance/             — vocabulary dominance and consistency CSVs")
print(f"  cross_class_summary.json     — all results in one file")
print(f"  README.md                    — interpretation guide")
print(f"\nDone at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
