import os
import json
import hashlib
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from datetime import datetime

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "leakage_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLITS = {
    "train": os.path.join(PROJECT_DIR, "datasets", "train.jsonl"),
    "eval" : os.path.join(PROJECT_DIR, "datasets", "eval.jsonl"),
    "test" : os.path.join(PROJECT_DIR, "datasets", "test.jsonl"),
}

# Similarity threshold — texts with Jaccard similarity above this are flagged
# 1.0 = exact duplicate, 0.8 = near-duplicate, 0.5 = loosely similar
THRESHOLD    = 0.8
NUM_PERM     = 128   # MinHash permutations — higher = more accurate
NGRAM_SIZE   = 5     # character n-gram size for shingling

print("=" * 70)
print("NEAR-DUPLICATE & LEAKAGE DETECTION")
print(f"  Method    : MinHash LSH (character {NGRAM_SIZE}-grams)")
print(f"  Threshold : Jaccard similarity >= {THRESHOLD}")
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

def make_minhash(text, num_perm=NUM_PERM, ngram=NGRAM_SIZE):
    m = MinHash(num_perm=num_perm)
    text = text.lower()
    for i in range(len(text) - ngram + 1):
        m.update(text[i:i+ngram].encode("utf-8"))
    return m

def exact_hash(text):
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

# ── Load all splits ───────────────────────────────────────────────────────────
data = {}
for name, path in SPLITS.items():
    texts, labels = load_split(path)
    data[name] = {"texts": texts, "labels": labels}
    print(f"\nLoaded {name:5s}: {len(texts)} samples")

# ── Step 1: Exact duplicate check within each split ──────────────────────────
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

# ── Step 3: Near-duplicate cross-split leakage (MinHash LSH) ─────────────────
print(f"\n{'─'*70}")
print(f"STEP 3 — Near-duplicate cross-split leakage (Jaccard >= {THRESHOLD})")
print(f"{'─'*70}")

near_leakage_summary = []

for split_a, split_b in pairs:
    print(f"\n  Checking {split_a} ↔ {split_b}...")
    texts_a = data[split_a]["texts"]
    texts_b = data[split_b]["texts"]

    # Build LSH index from split_a
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    print(f"    Building MinHash index for {split_a} ({len(texts_a)} samples)...")
    for i, text in enumerate(texts_a):
        m = make_minhash(text)
        lsh.insert(f"{split_a}_{i}", m)

    # Query with split_b
    print(f"    Querying with {split_b} ({len(texts_b)} samples)...")
    near_dupes = []
    for j, text in enumerate(texts_b):
        m = make_minhash(text)
        results = lsh.query(m)
        for match_key in results:
            i = int(match_key.split("_")[1])
            # Compute exact Jaccard for reporting
            m_a = make_minhash(texts_a[i])
            jaccard = m.jaccard(m_a)
            near_dupes.append({
                f"{split_a}_idx"    : i,
                f"{split_b}_idx"    : j,
                "jaccard_similarity": round(jaccard, 4),
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

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

summary = {
    "intra_split_exact_duplicates": intra_results,
    "cross_split_exact_leakage"   : exact_leakage_summary,
    "cross_split_near_duplicates" : near_leakage_summary,
}

print("\nIntra-split exact duplicates:")
for r in intra_results:
    status = "CLEAN" if r["exact_duplicates"] == 0 else "ISSUES FOUND"
    print(f"  {r['split']:5s}: {r['exact_duplicates']} duplicates — {status}")

print("\nCross-split exact leakage:")
for r in exact_leakage_summary:
    status = "CLEAN" if r["exact_leaks"] == 0 else "LEAKAGE FOUND"
    print(f"  {r['pair']:15s}: {r['exact_leaks']} exact matches — {status}")

print(f"\nCross-split near-duplicates (Jaccard >= {THRESHOLD}):")
for r in near_leakage_summary:
    status = "CLEAN" if r["near_duplicates"] == 0 else "NEAR-DUPLICATES FOUND"
    print(f"  {r['pair']:15s}: {r['near_duplicates']} pairs — {status}")

summary_path = os.path.join(OUTPUT_DIR, "leakage_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print(f"Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
