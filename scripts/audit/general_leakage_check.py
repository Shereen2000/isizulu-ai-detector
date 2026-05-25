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

THRESHOLD      = 0.8
NUM_PERM       = 128
NGRAM_SIZE     = 5
TOP_WORDS      = 50
MIN_VOCAB_FREQ = 10
CHI2_ALPHA     = 0.05
OUTPUT_DIR     = os.path.join(PROJECT_DIR, "general_leakage_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("GENERAL LEAKAGE & VOCABULARY DISTRIBUTION")
print(f"  Near-dup threshold : Jaccard >= {THRESHOLD}")
print(f"  N-grams            : character {NGRAM_SIZE}-grams")
print(f"  MinHash            : {NUM_PERM} permutations")
print(f"  Top words tracked  : {TOP_WORDS} (min freq={MIN_VOCAB_FREQ})")
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

def exact_hash(text):
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def word_frequencies(texts):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    return counter

# ── Load all splits ───────────────────────────────────────────────────────────
data = {}
for name, path in SPLITS.items():
    texts, labels = load_split(path)
    data[name] = {"texts": texts, "labels": labels}
    print(f"\nLoaded {name:5s}: {len(texts)} samples  "
          f"(label 0: {sum(1 for l in labels if l==0)}, "
          f"label 1: {sum(1 for l in labels if l==1)})")

# ── Step 1: Exact duplicates within each split ────────────────────────────────
print(f"\n{'─'*70}")
print("STEP 1 — Exact duplicates within each split")
print(f"{'─'*70}")

intra_results = []
for name in SPLITS:
    texts  = data[name]["texts"]
    labels = data[name]["labels"]
    seen, dupes = {}, []
    for i, text in enumerate(texts):
        h = exact_hash(text)
        if h in seen:
            dupes.append((seen[h], i, labels[seen[h]], labels[i]))
        else:
            seen[h] = i
    print(f"  {name:5s}: {len(dupes)} exact duplicates found")
    intra_results.append({"split": name, "exact_duplicates": len(dupes)})
    if dupes:
        rows = [{"split": name, "idx_a": a, "idx_b": b,
                 "label_a": la, "label_b": lb,
                 "same_label": la == lb} for a, b, la, lb in dupes]
        pd.DataFrame(rows).to_csv(
            os.path.join(OUTPUT_DIR, f"{name}_intra_duplicates.csv"), index=False)

# ── Step 2: Exact cross-split leakage ────────────────────────────────────────
print(f"\n{'─'*70}")
print("STEP 2 — Exact cross-split leakage")
print(f"{'─'*70}")

pairs = [("train", "eval"), ("train", "test"), ("eval", "test")]
exact_leakage_summary = []

for split_a, split_b in pairs:
    hashes_a = {exact_hash(t): i for i, t in enumerate(data[split_a]["texts"])}
    leaks = []
    for j, text in enumerate(data[split_b]["texts"]):
        h = exact_hash(text)
        if h in hashes_a:
            i = hashes_a[h]
            leaks.append({
                f"{split_a}_idx"  : i,
                f"{split_b}_idx"  : j,
                f"{split_a}_label": data[split_a]["labels"][i],
                f"{split_b}_label": data[split_b]["labels"][j],
                "label_match"     : data[split_a]["labels"][i] == data[split_b]["labels"][j],
                "text_snippet"    : text[:120],
            })
    print(f"  {split_a} ↔ {split_b}: {len(leaks)} exact matches")
    exact_leakage_summary.append({"pair": f"{split_a}↔{split_b}", "exact_leaks": len(leaks)})
    if leaks:
        pd.DataFrame(leaks).to_csv(
            os.path.join(OUTPUT_DIR, f"exact_leak_{split_a}_{split_b}.csv"), index=False)

# ── Step 3: Near-duplicate cross-split leakage ───────────────────────────────
print(f"\n{'─'*70}")
print(f"STEP 3 — Near-duplicate cross-split leakage (Jaccard >= {THRESHOLD})")
print(f"{'─'*70}")

near_leakage_summary = []

for split_a, split_b in pairs:
    print(f"\n  Checking {split_a} ↔ {split_b}...")
    texts_a = data[split_a]["texts"]
    texts_b = data[split_b]["texts"]

    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    for i, text in enumerate(texts_a):
        lsh.insert(f"{split_a}_{i}", make_minhash(text))

    near_dupes = []
    for j, text in enumerate(texts_b):
        m = make_minhash(text)
        for match_key in lsh.query(m):
            i       = int(match_key.split("_")[1])
            m_a     = make_minhash(texts_a[i])
            jaccard = round(m.jaccard(m_a), 4)
            if jaccard < THRESHOLD:
                continue
            near_dupes.append({
                f"{split_a}_idx"    : i,
                f"{split_b}_idx"    : j,
                "jaccard_similarity": jaccard,
                f"{split_a}_label"  : data[split_a]["labels"][i],
                f"{split_b}_label"  : data[split_b]["labels"][j],
                "label_match"       : data[split_a]["labels"][i] == data[split_b]["labels"][j],
                f"{split_a}_snippet": texts_a[i][:100],
                f"{split_b}_snippet": texts_b[j][:100],
            })

    n = len(near_dupes)
    print(f"    Near-duplicates found: {n}")
    near_leakage_summary.append({"pair": f"{split_a}↔{split_b}", "near_duplicates": n})

    if near_dupes:
        df = pd.DataFrame(near_dupes).sort_values("jaccard_similarity", ascending=False)
        df.to_csv(os.path.join(OUTPUT_DIR, f"near_leak_{split_a}_{split_b}.csv"), index=False)
        print(f"    Top 3 most similar pairs:")
        for _, row in df.head(3).iterrows():
            print(f"      Jaccard={row['jaccard_similarity']:.3f}  "
                  f"labels={row[f'{split_a}_label']}↔{row[f'{split_b}_label']}  "
                  f"match={row['label_match']}")

# ── Step 4: General vocabulary distribution across splits ─────────────────────
print(f"\n{'─'*70}")
print("STEP 4 — General vocabulary distribution (train → eval/test)")
print(f"{'─'*70}")

train_texts = data["train"]["texts"]
train_freq  = word_frequencies(train_texts)
n_train     = len(train_texts)

top_words = [w for w, c in train_freq.most_common(TOP_WORDS * 3) if c >= MIN_VOCAB_FREQ][:TOP_WORDS]
print(f"\n  Top {len(top_words)} words tracked from train (min freq={MIN_VOCAB_FREQ})")

vocab_rows = []
vocab_results = {}

for split_name in ["eval", "test"]:
    split_texts = data[split_name]["texts"]
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

        vocab_rows.append({
            "word"              : word,
            "compared_split"    : split_name,
            "train_count"       : tc,
            f"{split_name}_count": sc,
            "train_rate"        : train_rate,
            f"{split_name}_rate": split_rate,
            "rate_diff"         : round(abs(train_rate - split_rate), 6),
            "chi2"              : round(chi2, 4),
            "p_value"           : round(p, 6),
            "significant_shift" : p < CHI2_ALPHA,
        })

    sub     = [r for r in vocab_rows if r["compared_split"] == split_name]
    shifted = sum(1 for r in sub if r["significant_shift"])
    stable  = len(top_words) - shifted
    pct_stab = round(100 * stable / len(top_words), 1) if top_words else 0

    vocab_results[split_name] = {
        "stable"    : stable,
        "shifted"   : shifted,
        "pct_stable": f"{pct_stab}%",
    }
    print(f"  train↔{split_name}: {stable}/{len(top_words)} top words stable ({pct_stab}%), "
          f"{shifted} shifted significantly")

vocab_df = pd.DataFrame(vocab_rows)
vocab_df.to_csv(os.path.join(OUTPUT_DIR, "vocab_distribution.csv"), index=False)

# ── Save summary ──────────────────────────────────────────────────────────────
summary = {
    "near_dup_threshold"          : THRESHOLD,
    "ngram_size"                  : NGRAM_SIZE,
    "minhash_permutations"        : NUM_PERM,
    "vocab_top_words"             : TOP_WORDS,
    "vocab_min_freq"              : MIN_VOCAB_FREQ,
    "intra_split_exact_duplicates": intra_results,
    "cross_split_exact_leakage"   : exact_leakage_summary,
    "cross_split_near_duplicates" : near_leakage_summary,
    "vocab_distribution"          : vocab_results,
}

with open(os.path.join(OUTPUT_DIR, "leakage_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")

print("\nIntra-split exact duplicates:")
for r in intra_results:
    print(f"  {r['split']:5s}: {r['exact_duplicates']} — "
          f"{'CLEAN' if r['exact_duplicates'] == 0 else 'ISSUES FOUND'}")

print("\nCross-split exact leakage:")
for r in exact_leakage_summary:
    print(f"  {r['pair']:15s}: {r['exact_leaks']} — "
          f"{'CLEAN' if r['exact_leaks'] == 0 else 'LEAKAGE FOUND'}")

print(f"\nCross-split near-duplicates (Jaccard >= {THRESHOLD}):")
for r in near_leakage_summary:
    print(f"  {r['pair']:15s}: {r['near_duplicates']} — "
          f"{'CLEAN' if r['near_duplicates'] == 0 else 'NEAR-DUPLICATES FOUND'}")

print(f"\nVocabulary distribution stability (top {len(top_words)} train words):")
for split_name, res in vocab_results.items():
    print(f"  train↔{split_name}: {res['stable']}/{len(top_words)} stable ({res['pct_stable']})")

print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  *_intra_duplicates.csv    — exact duplicates within same split")
print(f"  exact_leak_*.csv          — exact cross-split leaks")
print(f"  near_leak_*.csv           — near-duplicate cross-split pairs")
print(f"  vocab_distribution.csv    — top train words tracked into eval/test")
print(f"  leakage_summary.json      — all results in one file")
print(f"\nDone at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
