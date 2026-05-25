import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
import pandas as pd
from datetime import datetime

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH  = os.path.join(PROJECT_DIR, "finetuned_classifier", "final_model")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "cleanlab_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLITS      = {
    "train": os.path.join(PROJECT_DIR, "datasets", "train.jsonl"),
    "eval" : os.path.join(PROJECT_DIR, "datasets", "eval.jsonl"),
    "test" : os.path.join(PROJECT_DIR, "datasets", "test.jsonl"),
}
MAX_LENGTH  = 512
BATCH_SIZE  = 16
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("CLEANLAB DATASET QUALITY AUDIT")
print("=" * 70)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print(f"   Device: {DEVICE}")

# ── Helper: load a split ──────────────────────────────────────────────────────
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
    return texts, np.array(labels)

# ── Helper: get predicted probabilities ──────────────────────────────────────
def get_probs(texts):
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = tokenizer(
                texts[i:i+BATCH_SIZE],
                truncation=True, padding=True,
                max_length=MAX_LENGTH, return_tensors="pt"
            ).to(DEVICE)
            logits = model(**batch).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            if (i // BATCH_SIZE) % 20 == 0:
                print(f"      {min(i+BATCH_SIZE, len(texts))}/{len(texts)} done")
    return np.concatenate(all_probs, axis=0)   # (n, 2)

# ── Audit each split ──────────────────────────────────────────────────────────
summary_rows = []

for split_name, split_path in SPLITS.items():
    print(f"\n{'─'*70}")
    print(f"Auditing: {split_name}.jsonl")
    print(f"{'─'*70}")

    texts, labels = load_split(split_path)
    print(f"   Samples : {len(texts)}  (label 0: {(labels==0).sum()}, label 1: {(labels==1).sum()})")
    print(f"   Running inference...")

    probs = get_probs(texts)

    # Cleanlab: find label issues
    label_issues_idx = find_label_issues(
        labels=labels,
        pred_probs=probs,
        return_indices_ranked_by="self_confidence",
    )

    # Cleanlab: score every sample (1.0 = clean, 0.0 = likely mislabeled)
    quality_scores = get_label_quality_scores(labels=labels, pred_probs=probs)

    n_issues = len(label_issues_idx)
    pct      = 100 * n_issues / len(texts)
    print(f"\n   Label issues found : {n_issues} / {len(texts)}  ({pct:.2f}%)")
    print(f"   Mean quality score : {quality_scores.mean():.4f}")
    print(f"   Min  quality score : {quality_scores.min():.4f}  (most suspicious sample)")

    # Build detailed dataframe for this split
    df = pd.DataFrame({
        "index"        : range(len(texts)),
        "given_label"  : labels,
        "given_label_name": ["human" if l == 0 else "machine" for l in labels],
        "prob_human"   : probs[:, 0].round(4),
        "prob_machine" : probs[:, 1].round(4),
        "quality_score": quality_scores.round(4),
        "flagged"      : [i in label_issues_idx for i in range(len(texts))],
        "text_snippet" : [t[:120] for t in texts],
    })

    # Save full results
    csv_path = os.path.join(OUTPUT_DIR, f"{split_name}_quality.csv")
    df.to_csv(csv_path, index=False)

    # Save flagged samples separately
    flagged_df = df[df["flagged"]].copy()
    flagged_path = os.path.join(OUTPUT_DIR, f"{split_name}_flagged.csv")
    flagged_df.to_csv(flagged_path, index=False)

    # Print top 5 most suspicious
    if n_issues > 0:
        print(f"\n   Top 5 most suspicious samples:")
        top5 = df[df["flagged"]].nsmallest(5, "quality_score")
        for _, row in top5.iterrows():
            print(f"      idx={row['index']:5d}  label={row['given_label_name']:7s}  "
                  f"p_human={row['prob_human']:.3f}  p_machine={row['prob_machine']:.3f}  "
                  f"quality={row['quality_score']:.3f}")
            print(f"      text: {row['text_snippet'][:80]}...")

    summary_rows.append({
        "split"             : split_name,
        "total_samples"     : len(texts),
        "label_issues"      : n_issues,
        "issue_pct"         : round(pct, 2),
        "mean_quality_score": round(float(quality_scores.mean()), 4),
        "min_quality_score" : round(float(quality_scores.min()), 4),
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

summary_path = os.path.join(OUTPUT_DIR, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary_rows, f, indent=2)

print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  <split>_quality.csv  — quality score for every sample")
print(f"  <split>_flagged.csv  — only the flagged/suspicious samples")
print(f"  summary.json         — overall stats per split")
print(f"\nDone at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
